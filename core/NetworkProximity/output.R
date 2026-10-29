# =============================================================================
# 05 OUTPUT — shaping results, not a modelling step.
#
# Two functions, both optional and both usable on their own:
#
#   compare_results()      join step 2 and step 3, flag what each mode supports
#   build_drug_contract()  collapse to one row per drug + attach drug structures,
#                          which is the join key to the virtual-screening component
#
# Source this after whichever steps you actually ran. compare_results() needs
# steps 2 and 3; build_drug_contract() needs step 2 and will take step 3 if you
# have it.
#
# Both functions accept the symbol-native `drug` column emitted by steps 2 and 4
# as well as the `drugbank_id` column the Entrez-era steps 1/3 emit, so the two
# halves of the component can be mixed without an external rename step.
# =============================================================================


# Accept either the symbol-native `drug` id column or the Entrez-era
# `drugbank_id`, and return a copy keyed on `drugbank_id`. The two halves of
# this component were migrated at different times (steps 0, 2 and 4 are symbol-native;
# steps 1, 3 and 5 still carry the Entrez-era names), and a silent column
# mismatch here is a merge that quietly drops every row.
.as_drugbank_id = function(x, what) {
  x = as.data.table(x)
  if ("drugbank_id" %in% names(x)) return(copy(x))
  if ("drug" %in% names(x)) { x = copy(x); setnames(x, "drug", "drugbank_id"); return(x) }
  stop(what, " needs a `drugbank_id` or `drug` column; got: ",
       paste(names(x), collapse = ", "))
}


# Cauchy combination of per-cell-type p-values (ACAT). Robust to the dependence
# between cell types, which a Fisher or Stouffer combination is not: the same
# drug is scored against overlapping graphs built from one dataset. Use when the
# question is "proximal in any or several cell types" rather than "proximal in
# its best cell type".
acat_p = function(p) {
  p = p[is.finite(p)]; if (!length(p)) return(NA_real_)
  p = pmin(pmax(p, 1e-12), 1 - 1e-12)
  0.5 - atan(mean(tan((0.5 - p) * pi))) / pi
}

#' Join static and dynamic results and flag supported drugs
#'
#' Join the static and dynamic results and flag drugs supported by each mode
#' (negative z, meaning closer than the matched null, at an FDR cutoff).
#'
#' @param static_result Output of [run_static_proximity()].
#' @param dynamic_result Output of [run_dynamic_proximity()].
#' @param gene_scores Output of [score_genes()] (reserved; kept for the
#'   established call signature).
#' @return data.table with one row per drug x cell type carrying both modes'
#'   statistics and the `static_supported` / `dynamic_supported` flags.
compare_results = function(static_result, dynamic_result, gene_scores) {
  s = .as_drugbank_id(static_result, "static_result")[, .(drugbank_id, static_z, static_p, static_fdr)]
  r = merge(.as_drugbank_id(dynamic_result, "dynamic_result"), s,
            by = "drugbank_id", all.x = TRUE)

  r[, static_supported  := !is.na(static_z) & static_z <= -1.96 & !is.na(static_fdr) & static_fdr <= 0.1]
  r[, dynamic_supported := !is.na(dyn_z)    & dyn_z    <= -1.96 & !is.na(dyn_fdr)    & dyn_fdr    <= 0.1]

  cols = c("drugbank_id", "drug_name", "cell_type",
           "static_z", "static_p", "static_fdr",
           "dyn_z", "dyn_p", "dyn_fdr",
           "dyn_null_mean", "dyn_null_sd", "dyn_n_null",
           "dyn_z_condnull", "dyn_p_condnull", "dyn_fdr_condnull",
           "cell_spec_weighted", "cond_score_weighted", "cond_spec_weighted", "signed_cond_weighted",
           "n_active_targets", "dominant_targets", "dominant_module_genes",
           "static_supported", "dynamic_supported")
  # keep only what the supplied tables actually carry: the optional condition
  # null is absent when no reference condition was given, and a caller may pass a
  # reduced dynamic table. Selecting a missing column is a hard error, and it is
  # not one worth having.
  r = r[, intersect(cols, names(r)), with = FALSE]
  setorder(r, dyn_z, static_z)
  r[]
}


#' Per-drug result contract, keyed by DrugBank ID
#'
#' Collapses the per-drug x cell-type results to one row per drug and attaches the
#' structure bridge, so the virtual-screening component's per-molecule binding scores can
#' be joined on `inchikey` / `smiles` without further wrangling.
#'
#' Ranks and percentiles are emitted alongside the z-scores deliberately. The two
#' components are not on a common scale -- the AI component's P(bind) is uncalibrated and its
#' spread depends on which model produced it -- so a fusion should join on ranks
#' or percentiles, not on raw scores.
#'
#' `has_structure` marks the drugs a structure-based model can score at all:
#' biotech drugs (antibodies and similar) carry no SMILES, so they are network-component
#' evidence only and must not be counted as AI-component misses.
#'
#' Taking the best cell type is a selection, and it is corrected here. A drug
#' tested in eight cell types gets eight draws at a low z; a drug tested in one
#' gets one, so an uncorrected `min(dyn_z)` rewards breadth of testing rather
#' than proximity. Three routes, all reported, one of which sets the flag:
#'
#'   - pre-specify the cell type(s) with `cell_types`. If the biology names the
#'     relevant context there is no selection left to correct, and this is both
#'     the cheapest and the most interpretable option;
#'   - `best_dyn_p_sidak`, the best p adjusted for the number of cell types
#'     tested, `1 - (1 - p)^K`. This sets `dynamic_supported` by default;
#'   - `dyn_p_combined`, a Cauchy combination over all cell types, for the
#'     question "proximal in any or several cell types".
#'
#' The Sidak route interacts with `n_perm`: at n_perm = 50 and K = 8 the
#' smallest attainable adjusted p is 1 - (1 - 1/51)^8 = 0.146, so nothing can
#' pass. At n_perm = 1000 it is 0.008.
#'
#' @param static_result Output of [run_static_proximity()] (step 2).
#' @param dynamic_result Output of [run_dynamic_proximity()] (step 3); optional.
#' @param structures Structure bridge; pass `drug_structures` from 00_setup.R.
#' @param fdr_cutoff FDR at which a drug counts as supported.
#' @param z_cutoff z at or below which a drug counts as proximal.
#' @param cell_type_selection How `dynamic_supported` handles the best-of-K
#'   selection: "sidak" (default), "cauchy", or "none" for the uncorrected
#'   per-cell-type FDR rule used before this correction existed.
#' @param cell_types Optional character vector of pre-specified cell types. When
#'   given, only these rows are considered and the multiplicity shrinks with them.
#' @param alpha Significance level for the selection-corrected p.
#' @return data.table, one row per `drugbank_id`, ordered by `static_rank`.
build_drug_contract = function(static_result, dynamic_result = NULL, structures,
                               fdr_cutoff = 0.1, z_cutoff = -1.96,
                               cell_type_selection = c("sidak", "cauchy", "none"),
                               cell_types = NULL, alpha = 0.05) {
  stopifnot("pass structures (drug_structures, see 00_setup.R)" = is.data.frame(structures))
  cell_type_selection = match.arg(cell_type_selection)
  sr = .as_drugbank_id(static_result, "static_result")
  keep = c("drugbank_id", "drug_name", "n_targets", "observed_distance",
           "static_z", "static_p", "static_fdr",
           if ("null_sd" %in% names(sr)) "null_sd")
  s = sr[, ..keep]
  s[, static_supported := !is.na(static_z) & static_z <= z_cutoff &
      !is.na(static_fdr) & static_fdr <= fdr_cutoff]

  # Degenerate null: when every permutation returns the same distance, null_sd is
  # 0 and z = (observed - mean)/0 = -Inf. That is a scale artifact, not evidence
  # of extreme proximity, and it slips past the usual guards -- is.na(-Inf) is
  # FALSE, so -Inf passes `z <= z_cutoff`, counts as supported, and takes rank 1
  # ahead of every finite z. It collapses as n_perm rises (no -Inf at all by 60
  # permutations on the real network), so this is a guard rail for small-n_perm
  # runs rather than a live defect. The statistic is left exactly as steps 2 and 3
  # compute it; the flag just makes the artifact visible.
  #
  # Both modes are covered: run_dynamic_proximity() divides by its own accumulated
  # sd the same way, and now reports it as `dyn_null_sd`. When that column is
  # present the dynamic flag is set from the sd directly; otherwise it falls back
  # to a non-finite, non-NA z -- which is the symptom that matters.
  if ("null_sd" %in% names(s)) {
    s[, static_null_degenerate := !is.na(null_sd) & null_sd == 0]
  } else {
    s[, static_null_degenerate := !is.na(static_z) & !is.finite(static_z)]
  }

  has_dyn = !is.null(dynamic_result) && nrow(dynamic_result) > 0
  if (has_dyn) {
    d = .as_drugbank_id(dynamic_result, "dynamic_result")[!is.na(dyn_z)]
    if (!is.null(cell_types)) {
      cell_types = as.character(cell_types)
      missing_ct = setdiff(cell_types, unique(d$cell_type))
      if (length(missing_ct))
        warning("cell_types not present in dynamic_result: ",
                paste(missing_ct, collapse = ", "), call. = FALSE)
      d = d[cell_type %in% cell_types]
    }
    has_dyn = nrow(d) > 0
  }
  if (has_dyn) {
    d[, .supported_ct := !is.na(dyn_z) & dyn_z <= z_cutoff &
        !is.na(dyn_fdr) & dyn_fdr <= fdr_cutoff]
    has_dyn_sd = "dyn_null_sd" %in% names(d)
    dsum = d[, {
      # Finite z only. A degenerate null (dyn_null_sd == 0) yields dyn_z = -Inf with an
      # empirical dyn_p of 1, and a bare which.min would hand it best_cell_type on no
      # evidence. With no finite z the selection is NA rather than an arbitrary row;
      # n_cell_types_tested still counts every cell type the drug was scored in.
      fin = which(is.finite(dyn_z))
      i = if (length(fin)) fin[which.min(dyn_z[fin])] else NA_integer_
      .(best_dyn_z = dyn_z[i],
        best_cell_type = cell_type[i],
        best_dyn_fdr = dyn_fdr[i],
        best_dyn_p = dyn_p[i],
        # Sidak over the number of cell types this drug was actually tested in,
        # which is the size of the selection that produced best_dyn_z.
        best_dyn_p_sidak = if (is.na(dyn_p[i])) NA_real_ else 1 - (1 - dyn_p[i])^.N,
        dyn_p_combined = acat_p(dyn_p),
        best_dyn_null_sd = if (has_dyn_sd) dyn_null_sd[i] else NA_real_,
        n_cell_types_tested = .N,
        n_cell_types_supported = sum(.supported_ct))
    }, by = drugbank_id]
    out = merge(s, dsum, by = "drugbank_id", all.x = TRUE)
  } else {
    out = copy(s)
    out[, `:=`(best_dyn_z = NA_real_, best_cell_type = NA_character_,
               best_dyn_fdr = NA_real_, best_dyn_p = NA_real_,
               best_dyn_p_sidak = NA_real_, dyn_p_combined = NA_real_,
               best_dyn_null_sd = NA_real_, n_cell_types_tested = 0L,
               n_cell_types_supported = 0L)]
  }
  out[is.na(n_cell_types_tested), n_cell_types_tested := 0L]
  out[is.na(n_cell_types_supported), n_cell_types_supported := 0L]

  # `n_cell_types_supported > 0` is the uncorrected rule: it counts a hit in any
  # of K cell types without paying for K. It stays reachable as "none" so the
  # pre-correction tables can be reproduced, but it is not the default.
  out[, dynamic_supported := switch(cell_type_selection,
    sidak  = !is.na(best_dyn_z) & best_dyn_z <= z_cutoff &
             !is.na(best_dyn_p_sidak) & best_dyn_p_sidak <= alpha,
    cauchy = !is.na(best_dyn_z) & best_dyn_z <= z_cutoff &
             !is.na(dyn_p_combined) & dyn_p_combined <= alpha,
    none   = n_cell_types_supported > 0)]
  setattr(out, "cell_type_selection", cell_type_selection)

  out[, dynamic_null_degenerate := if (all(is.na(best_dyn_null_sd))) {
        !is.na(best_dyn_z) & !is.finite(best_dyn_z)
      } else {
        (!is.na(best_dyn_null_sd) & best_dyn_null_sd == 0) |
        (!is.na(best_dyn_z) & !is.finite(best_dyn_z))
      }]
  out[, null_degenerate := static_null_degenerate | dynamic_null_degenerate]

  # rank 1 = most proximal (most negative z); an NA z never ranks
  out[, static_rank := frank(static_z, ties.method = "min", na.last = "keep")]
  out[, dynamic_rank := frank(best_dyn_z, ties.method = "min", na.last = "keep")]
  n_static = sum(!is.na(out$static_z)); n_dyn = sum(!is.na(out$best_dyn_z))
  out[, static_pct := if (n_static > 0) static_rank / n_static else NA_real_]
  out[, dynamic_pct := if (n_dyn > 0) dynamic_rank / n_dyn else NA_real_]

  st = .as_drugbank_id(structures, "structures")
  # load_dti() derives is_small_molecule from drug_type; older structure tables
  # may not carry it, so fall back to the same derivation rather than failing.
  if (!"is_small_molecule" %in% names(st))
    st[, is_small_molecule := .is_small_molecule(drug_type)]
  if (!"has_structure" %in% names(st))
    st[, has_structure := !is.na(smiles) & smiles != ""]
  st = st[, .(drugbank_id, drug_type, inchikey, smiles,
              is_small_molecule, has_structure)]
  out = merge(out, st, by = "drugbank_id", all.x = TRUE)
  out[is.na(is_small_molecule), is_small_molecule := FALSE]
  out[is.na(has_structure), has_structure := FALSE]

  setcolorder(out, c("drugbank_id", "drug_name", "drug_type",
                     "static_z", "static_fdr", "static_supported", "static_rank",
                     "best_dyn_z", "best_dyn_fdr", "best_dyn_p_sidak", "best_cell_type",
                     "dynamic_supported", "dynamic_rank", "null_degenerate"))
  out[, c("static_null_degenerate", "dynamic_null_degenerate") := NULL]
  setorder(out, static_rank, na.last = TRUE)
  out[]
}
