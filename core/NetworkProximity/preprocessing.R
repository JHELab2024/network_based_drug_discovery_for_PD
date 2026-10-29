# =============================================================================
# Section 1: preprocessing
#
# Turns the two user-facing inputs (a disease-module gene vector and single-cell
# data) into the gene_scores table that both proximity modes read from, plus the
# resolved analysis mode. The fixed PPI and DTI reference tables are internal
# package data (see data-raw/build_internal_data.R) and are not built here.
# =============================================================================

# Normalize a cell-type label to a compact token (lowercase, underscores).
clean_cell_type = function(x) {
  x = tolower(trimws(as.character(x)))
  x = gsub("[^a-z0-9]+", "_", x)
  x = gsub("^_|_$", "", x)
  x
}

# symbol -> Entrez. The supplied lookup table takes precedence. One symbol that
# maps to several Entrez IDs is ambiguous: drop it and record it. Many symbols
# that map to one Entrez is allowed and is collapsed by max() downstream.
map_symbols_to_entrez = function(symbols, symbol_to_entrez = NULL) {
  symbols = unique(as.character(symbols))
  if (!is.null(symbol_to_entrez)) {
    m = as.data.table(symbol_to_entrez)
    setnames(m, names(m)[1:2], c("symbol", "entrez_id"))
    m = unique(m[, .(symbol = as.character(symbol), entrez_id = as.character(entrez_id))])
    m = m[symbol %in% symbols & !is.na(entrez_id) & entrez_id != ""]
  } else {
    raw = AnnotationDbi::select(org.Hs.eg.db::org.Hs.eg.db, keys = symbols, keytype = "SYMBOL",
                                columns = c("SYMBOL", "ENTREZID"))
    m = as.data.table(raw)
    setnames(m, tolower(names(m)))
    m = m[!is.na(entrezid) & entrezid != "", .(symbol, entrez_id = as.character(entrezid))]
  }
  amb = m[, .N, by = symbol][N > 1, symbol]
  list(map = m[!symbol %in% amb], ambiguous = m[symbol %in% amb])
}

#' Resolve a disease-module gene vector to Entrez IDs present in the PPI
#'
#' Disease-module input: a plain character vector of Entrez IDs or gene symbols.
#' `id_type = "auto"` treats all-digit tokens as Entrez and everything else as
#' symbols. Returns the Entrez IDs that are present in the PPI, which is all the
#' static proximity and node-weight steps need.
#'
#' Note on weighting: earlier versions carried a per-gene genetic_score weight
#' (defaulting to 1). That is removed. Module genes now enter the model unweighted
#' at this stage; their influence is set later from the single-cell signal. A
#' weighted-vector interface can be reintroduced here if a genetic prior is wanted.
#'
#' @param genes Character vector of gene symbols or Entrez IDs (the disease module).
#' @param ppi_nodes Character vector of Entrez IDs in the PPI (defaults to the
#'   the `ppi_nodes` object from 00_setup.R).
#' @param symbol_to_entrez Optional two-column symbol/Entrez lookup; takes
#'   precedence over org.Hs.eg.db.
#' @param id_type One of "auto", "entrez", "symbol".
#' @return Character vector of unique Entrez IDs present in the PPI.
# prepare_module_genes = function(genes, ppi_nodes, symbol_to_entrez = NULL, id_type = "auto") {
#   genes = unique(trimws(as.character(genes)))
#   genes = genes[!is.na(genes) & genes != ""]
#   if (id_type == "auto") id_type = if (all(grepl("^[0-9]+$", genes))) "entrez" else "symbol"
#   if (id_type == "entrez") {
#     entrez = genes
#   } else {
#     entrez = map_symbols_to_entrez(genes, symbol_to_entrez)$map$entrez_id
#   }
#   unique(entrez[entrez %in% ppi_nodes])
# }

#' Single-cell to pseudobulk, per-cell-type DE and detection rates
#'
#' single-cell -> pseudobulk -> per-cell-type edgeR DE and detection rates.
#' `counts` is a genes x cells matrix; `meta` is a cell-level data.table.
#'
#' CONTRACT ON `de_table_symbol`: it carries `logfc` for EVERY tested gene at
#' every significance level, and nothing downstream may add an FDR filter here.
#' It now has two consumers. score_genes() folds it into the condition score,
#' where an FDR gate would be merely conservative -- but the directional metric
#' (RevERT) also takes its per-cell-type disease direction `s_g` from this table,
#' and there an FDR gate is destructive: per-cell-type pseudobulk DE with few
#' donors has low power, and the module-by-CMap-landmark overlap is already the
#' binding constraint at 5 to 18 genes per module. Take the sign of logfc for
#' every module gene and put confidence into the gene weight `w_g`, which the
#' score already supports. The only filter applied here is edgeR's
#' filterByExpr(), which removes genes with too few counts to model at all.
#'
#' @param counts Gene x cell count matrix (dense or sparse); rownames are symbols.
#' @param meta Cell-level data.table, one row per column of `counts`.
#' @param annotation_col,condition_col,donor_col Column names in `meta`.
#' @param condition_a Condition of interest (e.g. the disease level).
#' @param condition_b Optional reference condition. NULL disables the DE axis.
#' @param confounders Optional character vector of covariate columns in `meta`.
#' @param min_samples_per_group Minimum donors per group for a cell type to be tested.
#' @param min_cells_per_pseudobulk Minimum cells for a pseudobulk column to be used.
#' @return List with `pseudobulk`, `de_table_symbol` and `de_status`.
process_singlecell = function(counts, meta, annotation_col, condition_col, donor_col,
                              condition_a, condition_b = NULL, confounders = NULL,
                              min_samples_per_group = 2,
                              min_cells_per_pseudobulk = 5) {
  has_ref = !is.null(condition_b)

  meta[, ct_clean := clean_cell_type(get(annotation_col))]
  meta[, cond_raw := as.character(get(condition_col))]
  meta[, donor_raw := as.character(get(donor_col))]

  keep_cond  = if (has_ref) c(condition_a, condition_b) else condition_a
  keep_cells = meta$cond_raw %in% keep_cond
  meta   = meta[keep_cells]
  counts = counts[, keep_cells, drop = FALSE]

  # --- pseudobulk: donor x cell_type x condition ----------------------------
  meta[, group_id := paste(donor_raw, ct_clean, cond_raw, sep = "__")]
  gf = factor(meta$group_id, levels = unique(meta$group_id))
  design_sp = Matrix::sparse.model.matrix(~ 0 + gf)
  pb = counts %*% design_sp
  colnames(pb) = levels(gf)

  conf_cols = intersect(confounders, names(meta))
  pbm = unique(meta[, c("group_id", "donor_raw", "ct_clean", "cond_raw", conf_cols), with = FALSE])
  setnames(pbm, c("donor_raw", "ct_clean", "cond_raw"), c("donor", "cell_type", "condition"))
  pbm = pbm[match(colnames(pb), group_id)]
  pbm[, n_cells := as.integer(table(meta$group_id)[group_id])]
  pbm[, library_size := as.numeric(Matrix::colSums(pb))]

  cell_types = sort(unique(meta$ct_clean))

  # --- per-cell detection rates for the gates -------------------------------
  det_list = list()
  for (ct in cell_types) {
    in_ct = meta$ct_clean == ct
    ca = in_ct & meta$cond_raw == condition_a
    cb = if (has_ref) in_ct & meta$cond_raw == condition_b else rep(FALSE, nrow(meta))
    det_list[[ct]] = data.table(
      gene = rownames(counts), cell_type = ct,
      pct_expr_a = if (any(ca))    Matrix::rowMeans(counts[, ca,    drop = FALSE] > 0) else NA_real_,
      pct_expr_b = if (any(cb))    Matrix::rowMeans(counts[, cb,    drop = FALSE] > 0) else NA_real_,
      pct_expr_c = if (any(in_ct)) Matrix::rowMeans(counts[, in_ct, drop = FALSE] > 0) else NA_real_
    )
  }
  detection = rbindlist(det_list)

  # --- per cell type edgeR QL DE (only when a reference exists) --------------
  de_list = list(); status_list = list()
  for (ct in cell_types) {
    idx = which(pbm$cell_type == ct & pbm$condition %in% keep_cond &
                pbm$n_cells >= min_cells_per_pseudobulk)
    om = pbm[idx]; oc = pb[, idx, drop = FALSE]
    n_a = om[condition == condition_a, uniqueN(donor)]
    n_b = if (has_ref) om[condition == condition_b, uniqueN(donor)] else 0L
    st = data.table(cell_type = ct, n_donor_a = n_a, n_donor_b = n_b,
                    n_genes_tested = 0L, status = NA_character_)

    if (!has_ref) { st[, status := "no_reference"]; status_list[[ct]] = st; next }
    if (n_a < min_samples_per_group | n_b < min_samples_per_group) {
      st[, status := "skipped_too_few_samples"]; status_list[[ct]] = st; next
    }

    grp = factor(om$condition, levels = c(condition_b, condition_a))   # ref = baseline
    dge = edgeR::DGEList(counts = as.matrix(oc), group = grp)
    keep = edgeR::filterByExpr(dge, group = grp)
    if (sum(keep) < 20) { st[, status := "skipped_too_few_testable_genes"]; status_list[[ct]] = st; next }

    dge = dge[keep, , keep.lib.sizes = FALSE]
    dge = edgeR::calcNormFactors(dge)
    design = build_de_design(om, grp, conf_cols)
    grp_col = tail(grep("^group", colnames(design)), 1)

    de = tryCatch({
      dge = edgeR::estimateDisp(dge, design)
      fit = edgeR::glmQLFit(dge, design)
      qlf = edgeR::glmQLFTest(fit, coef = grp_col)
      # n = Inf: every testable gene, unfiltered. See the contract in the header.
      tt = as.data.table(edgeR::topTags(qlf, n = Inf)$table, keep.rownames = "gene")
      setnames(tt, c("logFC", "PValue", "FDR"), c("logfc", "pvalue", "fdr"))
      tt[, .(gene, cell_type = ct, logfc, pvalue, fdr, de_tested = TRUE)]
    }, error = function(e) NULL)

    if (!is.null(de)) {
      de_list[[ct]] = de
      st[, n_genes_tested := nrow(de)]; st[, status := "tested"]
    } else {
      st[, status := "de_failed"]
    }
    status_list[[ct]] = st
  }

  de_table_symbol = if (length(de_list))
    rbindlist(de_list, fill = TRUE) else
    data.table(gene = character(), cell_type = character(), logfc = numeric(),
               pvalue = numeric(), fdr = numeric(), de_tested = logical())

  list(
    pseudobulk = list(counts = pb, metadata = pbm, gene = rownames(pb), detection = detection),
    de_table_symbol = de_table_symbol,
    de_status = rbindlist(status_list, fill = TRUE)
  )
}

# Build the DE design matrix, adding usable confounders and falling back to the
# group-only design if the fuller model is rank-deficient.
build_de_design = function(om, grp, conf_cols) {
  df = data.frame(group = grp)
  ok = character()
  for (cf in conf_cols) {
    v = om[[cf]]
    if (is.null(v) || any(is.na(v)) || length(unique(v[!is.na(v)])) < 2) next
    df[[cf]] = v; ok = c(ok, cf)
  }
  form = if (length(ok)) as.formula(paste("~", paste(c(ok, "group"), collapse = " + "))) else ~ group
  design = model.matrix(form, data = df)
  if (qr(design)$rank < ncol(design) || (nrow(design) - ncol(design)) < 1)
    design = model.matrix(~ group, data = df)
  design
}

#' Per-gene, per-cell-type scores from detection, abundance and DE
#'
#' Collapse detection, abundance, and DE into per-gene per-cell-type scores.
#'
#' @param de_table_symbol DE table from [process_singlecell()].
#' @param pseudobulk Pseudobulk list from [process_singlecell()].
#' @param mode Mode spec from [resolve_mode()].
#' @param tau Detection-rate gate (fraction of cells expressing).
#' @param cap_quantile Quantile at which the condition score is capped, within cell type.
#' @param symbol_to_entrez Optional symbol/Entrez lookup. Pass
#'   [identity_symbol_map()] to run the component symbol-natively.
#' @param spec_measure How `cell_spec` is defined across cell types.
#'   "share" (default) is `abund / sum(abund)`, which reproduces prior runs but
#'   hands a uniformly expressed gene 1/K regardless of how strongly it is
#'   expressed, so the same gene scores 0.5 in a two-cell-type dataset and 0.1 in
#'   a ten-cell-type one. "relative" is `abund / max(abund)`, the ratio the tau
#'   specificity index is built from: it is 1 where the gene peaks, independent
#'   of K, and leaves a uniformly expressed gene usable as a target everywhere.
#'   Matters when K is large or the cell types are very imbalanced.
#' @param cond_gate Whether the condition score requires detection in "both"
#'   conditions (default, reproduces prior runs) or "either". Under "both" a gene
#'   switched on only in disease fails the gate and loses its condition score,
#'   which is the opposite of the intent.
#' @return data.table of per-gene per-cell-type scores; carries an
#'   `ambiguous_symbols` attribute.
score_genes = function(de_table_symbol, pseudobulk, mode, tau,
                       cap_quantile = 0.99, symbol_to_entrez = NULL,
                       spec_measure = c("share", "relative"),
                       cond_gate = c("both", "either")) {
  spec_measure = match.arg(spec_measure)
  cond_gate    = match.arg(cond_gate)
  counts = pseudobulk$counts
  meta   = pseudobulk$metadata
  detection = pseudobulk$detection
  cell_types = sort(unique(meta$cell_type))

  # --- abundance: pooled counts (cond_a + cond_b), column-normalised within ct
  ct_of_col = meta$cell_type[match(colnames(counts), meta$group_id)]
  pool = matrix(0, nrow(counts), length(cell_types),
                dimnames = list(rownames(counts), cell_types))
  for (ct in cell_types) {
    cols = which(ct_of_col == ct)
    if (length(cols)) pool[, ct] = Matrix::rowSums(counts[, cols, drop = FALSE])
  }
  csum = colSums(pool); csum[csum == 0] = 1
  abund_mat = sweep(pool, 2, csum, "/")

  sym = melt(as.data.table(abund_mat, keep.rownames = "symbol"),
             id.vars = "symbol", variable.name = "cell_type", value.name = "abund")
  sym[, cell_type := as.character(cell_type)]
  sym = merge(sym, detection, by.x = c("symbol", "cell_type"),
              by.y = c("gene", "cell_type"), all.x = TRUE)

  if (nrow(de_table_symbol)) {
    sym = merge(sym, de_table_symbol[, .(gene, cell_type, logfc, fdr)],
                by.x = c("symbol", "cell_type"), by.y = c("gene", "cell_type"), all.x = TRUE)
  } else {
    sym[, logfc := NA_real_]; sym[, fdr := NA_real_]
  }
  sym[is.na(fdr), fdr := 1]
  sym[is.na(logfc), logfc := 0]
  sym[, cond_raw := abs(logfc) * -log10(pmax(fdr, 1e-300))]
  sym[!is.finite(cond_raw), cond_raw := 0]

  # --- symbol -> Entrez (drop one-to-many, collapse many-to-one by max) ------
  mapres = map_symbols_to_entrez(unique(sym$symbol), symbol_to_entrez)
  sym = merge(sym, mapres$map, by = "symbol")

  g = sym[, .(
    symbol     = symbol[which.max(abund + cond_raw)],
    abund      = max(abund),
    cond_raw   = max(cond_raw),
    pct_expr_a = suppressWarnings(max(pct_expr_a, na.rm = TRUE)),
    pct_expr_b = suppressWarnings(max(pct_expr_b, na.rm = TRUE)),
    pct_expr_c = suppressWarnings(max(pct_expr_c, na.rm = TRUE)),
    logfc      = logfc[which.max(cond_raw)],
    fdr        = min(fdr)
  ), by = .(entrez_id, cell_type)]
  for (cc in c("pct_expr_a", "pct_expr_b", "pct_expr_c"))
    g[!is.finite(get(cc)), (cc) := NA_real_]

  # --- condition_score (within ct, 99th-pct capped) -------------------------
  g[, cap99 := {
    pos = cond_raw[is.finite(cond_raw) & cond_raw > 0]
    cap = if (length(pos)) as.numeric(quantile(pos, cap_quantile, na.rm = TRUE)) else 1
    if (!is.finite(cap) || cap <= 0) 1 else cap
  }, by = cell_type]
  g[, cond := pmin(cond_raw / cap99, 1)]
  g[!is.finite(cond), cond := 0]

  # --- gates ----------------------------------------------------------------
  g[, gate_cond := if (cond_gate == "both") {
        !is.na(pct_expr_a) & pct_expr_a >= tau &
        !is.na(pct_expr_b) & pct_expr_b >= tau
      } else {
        (!is.na(pct_expr_a) & pct_expr_a >= tau) |
        (!is.na(pct_expr_b) & pct_expr_b >= tau)
      }]
  g[, gate_node := !is.na(pct_expr_c) & pct_expr_c >= tau]

  if (!isTRUE(mode$axes_active$cond)) g[, gate_cond := FALSE]
  g[gate_cond == FALSE, cond := 0]                       # condition gate suppresses cond
  g[, signed_cond := sign(logfc) * cond]                 # report only

  # --- cross-cell-type partitions -------------------------------------------
  # cell_spec enters the node weights, so its scale matters. "share" divides by
  # the sum over cell types and therefore depends on K; "relative" divides by the
  # gene's own maximum and does not. Both give 1 when K = 1.
  g[, spec_denom := if (spec_measure == "share") sum(abund) else max(abund), by = entrez_id]
  g[, cell_spec := ifelse(spec_denom > 0, abund / spec_denom, 0)]   # K=1 -> 1; enters node weights
  g[, cspec_denom := sum(cond), by = entrez_id]
  g[, cond_spec := ifelse(cspec_denom > 0, cond / cspec_denom, 0)]  # report only

  attr(g, "spec_measure") = spec_measure
  attr(g, "cond_gate")    = cond_gate
  gene_scores = g[, .(entrez_id, symbol, cell_type,
                      abund, cond, signed_cond, cell_spec, cond_spec,
                      pct_expr_a, pct_expr_b, pct_expr_c,
                      gate_node, gate_cond, logfc, fdr, cond_raw, cap99)]
  attr(gene_scores, "ambiguous_symbols") = mapres$ambiguous
  attr(gene_scores, "spec_measure") = spec_measure
  attr(gene_scores, "cond_gate")    = cond_gate
  gene_scores[]
}

#' Resolve the active analysis axes from the data shape
#'
#' Resolve which analysis axes are active from the data shape, and set the
#' convex-combination lambdas for the module-gene and drug-target node weights.
#'
#' @param n_cell_types Number of cell types present after preprocessing.
#' @param has_ref TRUE when a reference condition was supplied.
#' @return List describing the mode: label, active axes, lambdas, null bins.
resolve_mode = function(n_cell_types, has_ref) {
  multi_ct = n_cell_types >= 2
  spec_on = multi_ct; cond_on = has_ref
  mode_label =
    if (spec_on && cond_on) "specificity_x_condition" else
    if (spec_on && !cond_on) "specificity_only" else
    if (!spec_on && cond_on) "condition_only" else "static_baseline_only"

  renorm = function(s, c, as_, ac_) {
    s2 = if (as_) s else 0; c2 = if (ac_) c else 0
    tot = s2 + c2
    if (tot <= 0) c(spec = 0, cond = 0) else c(spec = s2 / tot, cond = c2 / tot)
  }

  list(
    mode = mode_label,
    axes_active = list(spec = spec_on, cond = cond_on),
    lambda_module = renorm(0.5, 0.5, spec_on, cond_on),   # module-gene anchors
    lambda_target = renorm(0.7, 0.3, spec_on, cond_on),   # drug-target anchors
    null_bins = c("degree", "abundance", if (multi_ct) "specificity" else NULL),
    run_cond_null = cond_on
  )
}
