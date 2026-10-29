# =============================================================================
# NetworkProximity : combination.R
#
# Network-based drug-combination screen (Cheng et al. 2019). For each unordered
# pair of drugs it computes the target-set separation
#
#     s_ab = d_ab - (d_aa + d_bb) / 2
#
# and classifies the pair into one of six network relationships P1-P6 from
# whether each drug is close to the module and whether their targets are
# separated. P2 (complementary exposure: both drugs close, targets separated)
# is the nomination class.
#
# Symbol-native: drug targets are gene symbols. The drug identifier stays a
# stable drug id (e.g. a DrugBank id).
#
# The set-distance helpers (.closest_within_set, .closest_between_sets and the
# one-shot distance matrix) are shared with the FunctionalDirection component and now
# live beside it in NetworkProximity/set_distance.R.
# =============================================================================

suppressPackageStartupMessages({ library(data.table); library(igraph) })

# The set-distance helpers live beside this file, in NetworkProximity/set_distance.R.
# FunctionalDirection keeps its own copy of the same 3.7 kB file rather than reaching
# across components, so each component is self-contained and can be dropped into a project alone.
# Auto-located when this file is source()d by path; otherwise source
# NetworkProximity/set_distance.R first, or set options(dimer.model_root = ...).
if (!exists(".closest_between_sets", mode = "function")) {
  .np_find_helper <- function() {
    for (i in seq_len(sys.nframe())) {
      of <- sys.frame(i)$ofile
      if (!is.null(of))
        return(file.path(dirname(normalizePath(of, mustWork = FALSE)), "set_distance.R"))
    }
    root <- getOption("dimer.model_root")
    if (is.null(root)) NULL else file.path(root, "NetworkProximity", "set_distance.R")
  }
  .np_helper <- .np_find_helper()
  if (!is.null(.np_helper) && file.exists(.np_helper)) source(.np_helper) else
    stop("cannot locate NetworkProximity/set_distance.R; source() it first or set ",
         "options(dimer.model_root = '/path/to/model_release/core')")
  rm(.np_helper, .np_find_helper)
}


# Score drug pairs for complementary exposure.
#   module_genes : character vector of module gene symbols
#   drug_list    : character vector of drug ids, or a data.frame with a `drug` column
#   ppi_graph    : igraph PPI (node names = gene symbols)
#   dti          : data.table with columns  drug, drug_name, target_gene
#   blacklist    : adverse-pair table with columns drug_a, drug_b. Required; pass
#                  NULL to deliberately score without the safety filter.
#   drug_scores  : optional run_static_proximity() result to reuse (avoids a second
#                  permutation loop and keeps one statistical convention).
#   n_perm, seed : used only when drug_scores is not supplied.
#   z_cutoff     : a drug counts as close to the module when its z is below this
#                  (Cheng convention 0; tighten to -1.96 for significance).
# Returns a list with `pairs`, `drug_scores`, `drug_targets`,
# `removed_by_blacklist` and `report`.
run_drug_combination = function(module_genes, drug_list, ppi_graph, dti, blacklist,
                                drug_scores = NULL, n_perm = 1000, seed = 1,
                                z_cutoff = 0) {
  stopifnot("pass an igraph ppi_graph" = inherits(ppi_graph, "igraph"),
            "pass a data.frame dti"    = is.data.frame(dti))
  if (missing(blacklist))
    stop("`blacklist` is required: pass the adverse-pair table, or NULL to ",
         "deliberately score without the safety filter")

  # --- normalise the drug list --------------------------------------------
  if (is.data.frame(drug_list)) {
    stopifnot("drug_list needs a `drug` column" = "drug" %in% names(drug_list))
    drug_ids_in = as.character(drug_list$drug)
  } else {
    drug_ids_in = as.character(drug_list)
  }
  drug_ids_in = unique(trimws(drug_ids_in))
  drug_ids_in = drug_ids_in[!is.na(drug_ids_in) & drug_ids_in != ""]
  if (length(drug_ids_in) < 2)
    stop("run_drug_combination() needs at least 2 drugs; got ", length(drug_ids_in))

  ppi_nodes_v = igraph::V(ppi_graph)$name
  module_genes = unique(as.character(module_genes))
  module_genes = module_genes[module_genes %in% ppi_nodes_v]
  if (length(module_genes) < 2)
    stop("fewer than 2 module genes are present in the PPI")

  # --- drug -> targets (in PPI) -------------------------------------------
  d = unique(dti[, .(drug = trimws(as.character(drug)),
                     drug_name = as.character(drug_name),
                     target_gene = as.character(target_gene))])
  d = d[drug %in% drug_ids_in & target_gene %in% ppi_nodes_v]
  drug_name_map = d[, .(drug_name = drug_name[!is.na(drug_name) & drug_name != ""][1]), by = drug]
  drug_targets = lapply(split(d$target_gene, d$drug), function(x) sort(unique(x)))
  drug_targets = drug_targets[vapply(drug_targets, length, integer(1)) > 0]
  usable = names(drug_targets)
  if (length(usable) < 2)
    stop("fewer than 2 of the supplied drugs have targets in the PPI (got ", length(usable), ")")

  # --- drug -> module proximity: reuse static proximity when available -----
  if (is.null(drug_scores)) {
    drug_scores = run_static_proximity(ppi_graph = ppi_graph, module_genes = module_genes,
                                       dti = d, n_perm = n_perm, seed = seed)
    proximity_source = sprintf("computed here (n_perm = %d)", n_perm)
  } else {
    drug_scores = as.data.table(drug_scores)
    proximity_source = "supplied (reused from run_static_proximity)"
  }
  ds = drug_scores[drug %in% usable, .(drug, static_z, static_p, static_fdr)]
  ds[, direct_module_targets := vapply(drug, function(x)
        length(intersect(drug_targets[[x]], module_genes)), integer(1))]

  # --- one distance computation for every target involved ------------------
  dmat = target_distance_matrix(ppi_graph, unlist(drug_targets, use.names = FALSE))

  # --- unordered pairs, blacklist removed ----------------------------------
  pair_idx = utils::combn(sort(usable), 2)
  pairs = data.table(drug_a = pair_idx[1, ], drug_b = pair_idx[2, ])
  pairs[, pair_key := paste(drug_a, drug_b, sep = "__")]   # drug_a < drug_b by construction

  removed = pairs[0]
  if (!is.null(blacklist) && nrow(blacklist) > 0) {
    bl = as.data.table(blacklist)
    stopifnot("blacklist needs columns drug_a and drug_b" =
                all(c("drug_a", "drug_b") %in% names(bl)))
    a = trimws(as.character(bl$drug_a)); b = trimws(as.character(bl$drug_b))
    bl_keys = unique(ifelse(a < b, paste(a, b, sep = "__"), paste(b, a, sep = "__")))
    removed = pairs[pair_key %in% bl_keys]
    pairs = pairs[!pair_key %in% bl_keys]
  }
  if (nrow(pairs) == 0)
    stop("no drug pairs left to score after blacklist removal")

  # --- separation ----------------------------------------------------------
  within = vapply(usable, function(x) .closest_within_set(drug_targets[[x]], dmat), numeric(1))
  pairs[, d_aa := within[drug_a]]
  pairs[, d_bb := within[drug_b]]
  pairs[, d_ab := vapply(seq_len(.N), function(i)
        .closest_between_sets(drug_targets[[drug_a[i]]], drug_targets[[drug_b[i]]], dmat), numeric(1))]
  pairs[, s_ab := d_ab - (d_aa + d_bb) / 2]

  pairs[, n_targets_a := vapply(drug_a, function(x) length(drug_targets[[x]]), integer(1))]
  pairs[, n_targets_b := vapply(drug_b, function(x) length(drug_targets[[x]]), integer(1))]
  pairs[, n_shared_targets := vapply(seq_len(.N), function(i)
        length(intersect(drug_targets[[drug_a[i]]], drug_targets[[drug_b[i]]])), integer(1))]
  pairs[, jaccard_targets := vapply(seq_len(.N), function(i)
        length(intersect(drug_targets[[drug_a[i]]], drug_targets[[drug_b[i]]])) /
        length(union(drug_targets[[drug_a[i]]], drug_targets[[drug_b[i]]])), numeric(1))]

  # --- attach per-drug proximity ------------------------------------------
  pairs = merge(pairs, ds[, .(drug_a = drug, z_a = static_z, p_a = static_p,
                              fdr_a = static_fdr, direct_module_targets_a = direct_module_targets)],
                by = "drug_a", all.x = TRUE)
  pairs = merge(pairs, ds[, .(drug_b = drug, z_b = static_z, p_b = static_p,
                              fdr_b = static_fdr, direct_module_targets_b = direct_module_targets)],
                by = "drug_b", all.x = TRUE)
  pairs = merge(pairs, drug_name_map[, .(drug_a = drug, drug_name_a = drug_name)], by = "drug_a", all.x = TRUE)
  pairs = merge(pairs, drug_name_map[, .(drug_b = drug, drug_name_b = drug_name)], by = "drug_b", all.x = TRUE)

  # --- classify (NA-safe: an unscored drug must not silently become "close") -
  pairs[, both_close := !is.na(z_a) & !is.na(z_b) & z_a < z_cutoff & z_b < z_cutoff]
  pairs[, one_close  := !is.na(z_a) & !is.na(z_b) & xor(z_a < z_cutoff, z_b < z_cutoff)]
  pairs[, targets_separated := !is.na(s_ab) & s_ab >= 0]
  pairs[, mean_drug_z := rowMeans(cbind(z_a, z_b), na.rm = TRUE)]
  pairs[, max_drug_fdr := vapply(seq_len(.N), function(i)
        { v = c(fdr_a[i], fdr_b[i]); v = v[!is.na(v)]; if (!length(v)) NA_real_ else max(v) }, numeric(1))]

  pairs[, network_class := "unclassified"]
  pairs[both_close & !targets_separated, network_class := "P1_overlapping_exposure"]
  pairs[both_close &  targets_separated, network_class := "P2_complementary_exposure"]
  pairs[one_close  & !targets_separated, network_class := "P3_indirect_exposure"]
  pairs[one_close  &  targets_separated, network_class := "P4_single_exposure"]
  pairs[!both_close & !one_close & !targets_separated & !is.na(s_ab), network_class := "P5_non_exposure"]
  pairs[!both_close & !one_close &  targets_separated, network_class := "P6_independent_action"]

  pairs[, class_rank := 9L]
  pairs[network_class == "P2_complementary_exposure", class_rank := 1L]
  pairs[network_class == "P1_overlapping_exposure",   class_rank := 2L]
  pairs[network_class == "P4_single_exposure",        class_rank := 3L]
  pairs[network_class == "P3_indirect_exposure",      class_rank := 4L]
  pairs[network_class == "P6_independent_action",     class_rank := 5L]
  pairs[network_class == "P5_non_exposure",           class_rank := 6L]

  setorder(pairs, class_rank, mean_drug_z, -s_ab, max_drug_fdr)
  pairs[, overall_rank := seq_len(.N)]
  setcolorder(pairs, c("drug_a", "drug_b", "drug_name_a", "drug_name_b",
                       "network_class", "class_rank", "overall_rank",
                       "s_ab", "d_ab", "d_aa", "d_bb", "z_a", "z_b"))

  list(pairs = pairs[], drug_scores = ds[], drug_targets = drug_targets,
       removed_by_blacklist = removed[],
       report = list(
         drugs_supplied = length(drug_ids_in),
         drugs_with_ppi_targets = length(usable),
         drugs_dropped = setdiff(drug_ids_in, usable),
         module_genes_in_ppi = length(module_genes),
         unordered_pairs_possible = length(usable) * (length(usable) - 1) / 2,
         pairs_removed_by_blacklist = nrow(removed),
         pairs_scored = nrow(pairs),
         proximity_source = proximity_source,
         z_cutoff = z_cutoff,
         class_counts = as.list(table(pairs$network_class))))
}


# Filter a run_drug_combination() result to the nomination class: both drugs
# closer to the module than their matched nulls, with separated target sets.
nominate_complementary_pairs = function(combination, max_fdr = NULL, min_s_ab = 0, n = NULL) {
  pairs = if (is.data.frame(combination)) as.data.table(combination) else combination$pairs
  out = pairs[network_class == "P2_complementary_exposure" & !is.na(s_ab) & s_ab >= min_s_ab]
  if (!is.null(max_fdr)) out = out[!is.na(max_drug_fdr) & max_drug_fdr <= max_fdr]
  setorder(out, mean_drug_z, -s_ab)
  if (!is.null(n)) out = out[seq_len(min(n, nrow(out)))]
  out[]
}
