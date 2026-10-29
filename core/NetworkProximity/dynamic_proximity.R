# =============================================================================
# Section 3: context-specific dynamic proximity
#
# Context-specific proximity on a per-cell-type PPI: an edge is kept when both
# endpoints are detected in that cell type (`pct_expr_c >= tau`), so paths can
# only run through genes the cell type actually expresses. Path length is hop
# count on this filtered graph; expression enters as a tie-break between
# equal-length paths and, materially, through the target weights
# (`0.7 * specificity + 0.3 * condition`) in the per-drug weighted mean and
# through which module genes qualify as anchors. The null is matched on degree,
# abundance, specificity and condition bins.
#
# WHY THAT WORDING, AND NOT "weighted by abundance x abundance". Under the
# default `edge_weight_mode = "legacy_log"` the edge length is
# -log(abund_from * abund_to + epsilon) with epsilon = 1e-6, while `abund` is a
# gene's share of the cell type's pooled counts (~1e-5 to 1e-4 for a typical
# gene). The product of two shares is therefore several orders of magnitude
# below epsilon for nearly every edge, so epsilon sets the length:
#
#     both endpoints at ~100 CPM   -> product 1e-8  -> length 13.81
#     both endpoints at ~1000 CPM  -> product 1e-6  -> length 13.12
#     both endpoints at 1% of counts -> product 1e-4 -> length  9.20
#     undetected / low             -> << 1e-6       -> length 13.82
#
# The module-anchor penalty in calc_module_distance_vector() spans only 0 to
# ~6.91 (at w/max = 0.001) from the other side, so two short hops (2 x 9.2) are
# always longer than one long hop (13.8). Hop count decides every shortest path
# and expression only breaks ties. That is a defensible model and it is the one
# that produced the reported results, so it stays the default -- but it must be
# described as what it is.
#
# `edge_weight_mode` makes the alternatives available without changing the
# default. Do not switch it in the same run that regenerates the single-cell
# checkpoints: a changed result would then have two possible causes.
#
#   "legacy_log" (default) -log(abund * abund + epsilon); reproduces prior runs
#   "hop"                  every edge length 1, anchor penalty 0.99*(1-ws), as
#                          the revision plan specifies. Numbers read as hops and
#                          the ranking is preserved to within permutation noise
#                          (measured Spearman 0.96-0.98), but not exactly
#   "hop_exact"            every edge length 1, anchor penalty divided by
#                          -log(epsilon). A uniform rescale of legacy_log, so the
#                          ranking is preserved EXACTLY (measured 1.000) while the
#                          numbers read as hops
#   "bfs"                  unweighted breadth-first search; drops the tie-break
#                          entirely and is several times faster per permutation,
#                          which is what pays for a higher n_perm
#   "bounded"              1 + lambda * (1 - sqrt(pct_from * pct_to)); detection
#                          fractions are already on [0, 1], so a hop stays a hop
#                          and expression modulates within a bounded band. Use
#                          only if a claim depends on weighted paths, and report
#                          a 3-value sensitivity over lambda.
# =============================================================================

# Weighted distance from the whole module to every node in one Dijkstra run:
# wire one virtual source to every module anchor, giving higher-weight anchors
# smaller source penalties, then take shortest paths from that source.
calc_module_distance_vector = function(graph, module_weights, nodes, epsilon = 1e-6,
                                       edge_weight_mode = c("legacy_log", "hop", "hop_exact",
                                                            "bfs", "bounded"),
                                       anchor_scale = NULL) {
  edge_weight_mode = match.arg(edge_weight_mode)
  # How much of one hop the anchor penalty is allowed to span. This is the whole
  # tie-break, so it decides how much the module weights can reorder drugs.
  if (is.null(anchor_scale))
    anchor_scale = switch(edge_weight_mode, hop = 0.99, bounded = 0.5, 0)
  module_weights = module_weights[is.finite(module_weights) & module_weights > 0]
  module_weights = module_weights[names(module_weights) %in% igraph::V(graph)$name]
  if (length(module_weights) < 2) return(setNames(rep(NA_real_, length(nodes)), nodes))

  source_name = "__module_source__"
  ws = module_weights / max(module_weights)

  # Source penalty per anchor.
  #
  # ANCHOR PENALTY, AND WHY "hop" IS ALMOST BUT NOT EXACTLY THE SAME MODEL.
  # legacy_log's penalty is logarithmic in ws; the revision plan's B3a proposes
  # the linear 0.99 * (1 - ws). In hop units (one legacy hop = -log(epsilon) =
  # 13.82) they differ most exactly where most anchors sit:
  #
  #     w/max     legacy penalty        "hop" penalty
  #     1         0.00 hops             0.00 hops
  #     0.1       0.17 hops             0.89 hops     <- 5x
  #     0.01      0.33 hops             0.98 hops     <- 3x
  #     0.001     0.50 hops             0.99 hops
  #
  # In practice that barely moves the ranking, because the penalty is a
  # tie-break against a whole hop either way. Measured on synthetic data in the
  # real abundance regime (w_edge four orders below epsilon, n_perm = 200), the
  # Spearman of dyn_z against legacy_log was 0.96 to 0.98 per cell type, against
  # a legacy-vs-legacy seed control of 0.87 to 1.00. So B3a's acceptance bar is
  # met to within permutation noise, but it is an approximation, not an identity.
  #
  # "hop_exact" IS an identity. Dividing the legacy penalty by -log(epsilon) --
  # which is the legacy cost of one hop once w_edge falls below epsilon -- scales
  # the entire metric by a constant. Shortest paths are unchanged, every distance
  # is the legacy distance over 13.82, and z is invariant under a uniform rescale
  # because the null rescales with it. Measured Spearman: 1.000 in every cell
  # type. Use this one if the point of switching is to fix the units and change
  # nothing else.
  #
  # RUNNING THE ACCEPTANCE CHECK: drop non-finite z first. A degenerate null
  # (null_sd == 0) gives z = -Inf, and leaving those in a rank correlation makes
  # it report which drugs blew up rather than whether the orderings agree. With
  # them included the same comparison reads 0.61 to 0.76 and looks like a real
  # reordering; it is not. Use legacy-vs-legacy at two seeds as the control.
  pen = switch(edge_weight_mode,
    legacy_log = { p = -log(pmin(ws, 1 - epsilon) + epsilon); p[p < 0] = 0; p },
    hop_exact  = { p = -log(pmin(ws, 1 - epsilon) + epsilon) / (-log(epsilon))
                   p[p < 0] = 0; p },
    hop        = anchor_scale * (1 - ws),
    bounded    = anchor_scale * (1 - ws),
    bfs        = rep(0, length(ws)))

  g2 = igraph::add_vertices(graph, 1, name = source_name)
  ep = as.vector(rbind(rep(source_name, length(module_weights)), names(module_weights)))
  oc = igraph::ecount(g2)
  g2 = igraph::add_edges(g2, ep)
  ne = seq.int(oc + 1, igraph::ecount(g2))
  ed = igraph::edge_attr(g2, "edge_distance")
  if (is.null(ed)) ed = rep(NA_real_, igraph::ecount(g2))
  ed[ne] = pen
  igraph::E(g2)$edge_distance = ed

  # bfs: unweighted search, so the virtual source costs exactly one hop and is
  # subtracted back off. Anchor weights then decide only WHICH module genes are
  # anchors, which is already what they do in practice under legacy_log.
  dv = if (edge_weight_mode == "bfs") {
    as.numeric(igraph::distances(g2, v = source_name, to = nodes, weights = NA)) - 1
  } else {
    as.numeric(igraph::distances(g2, v = source_name, to = nodes,
                                 weights = igraph::E(g2)$edge_distance))
  }
  names(dv) = nodes
  dv[is.infinite(dv) | dv < 0] = NA_real_
  dv
}

# Rank-bin a numeric vector into n_bins (missing/non-finite treated as 0).
make_rank_bins = function(x, n_bins = 20) {
  x[is.na(x) | !is.finite(x)] = 0
  if (length(unique(x)) <= 1) return(rep("bin_1", length(x)))
  ranks = data.table::frank(x, ties.method = "average")
  bins = ceiling(ranks / max(ranks) * n_bins)
  bins[bins < 1] = 1
  paste0("bin_", bins)
}

# Precompute nested candidate lists for matched sampling: for each prefix of the
# match columns, a map from the joint bin key to the genes carrying that key.
prepare_sampling_index = function(bin_table, match_cols) {
  index_list = list()
  for (n_col in seq_along(match_cols)) {
    cols_now = match_cols[seq_len(n_col)]
    key_vector = do.call(paste, c(bin_table[, ..cols_now], sep = "|"))
    index_list[[n_col]] = list(
      candidate_list = split(bin_table$entrez_id, key_vector),
      gene_key = setNames(key_vector, bin_table$entrez_id)
    )
  }
  list(
    index_list = index_list,
    degree_list = split(bin_table$entrez_id, bin_table$degree_bin),
    gene_degree = setNames(bin_table$degree_bin, bin_table$entrez_id),
    all_genes = bin_table$entrez_id
  )
}

# For each real gene draw one matched gene, backing off from the fullest joint
# bin to coarser keys, then to degree only, then to any unused gene.
sample_matched_genes = function(real_genes, sampling_index) {
  real_genes = as.character(real_genes)
  sampled = character()
  for (one_gene in real_genes) {
    chosen = NA_character_
    for (index_i in rev(seq_along(sampling_index$index_list))) {
      gene_key = sampling_index$index_list[[index_i]]$gene_key[[one_gene]]
      if (!is.null(gene_key) && !is.na(gene_key)) {
        candidates = sampling_index$index_list[[index_i]]$candidate_list[[gene_key]]
        candidates = setdiff(candidates, sampled)
        if (length(candidates) > 0) { chosen = sample(candidates, 1); break }
      }
    }
    if (is.na(chosen)) {
      degree_key = sampling_index$gene_degree[[one_gene]]
      degree_candidates = setdiff(sampling_index$degree_list[[degree_key]], sampled)
      if (length(degree_candidates) > 0) chosen = sample(degree_candidates, 1)
    }
    if (is.na(chosen)) {
      all_candidates = setdiff(sampling_index$all_genes, sampled)
      if (length(all_candidates) == 0) all_candidates = sampling_index$all_genes
      chosen = sample(all_candidates, 1)
    }
    sampled = c(sampled, chosen)
  }
  sampled
}

#' Build one detection-filtered PPI graph per cell type
#'
#' One PPI graph per cell type: keep only edges whose endpoints are BOTH
#' detected in that cell type (`pct_expr_c >= tau`), so paths can only run
#' through genes the cell type actually expresses. That gate is what makes the
#' graph context specific. The edge length is then set by `edge_weight_mode`;
#' under the default it is nominally an abundance transform but is in practice a
#' constant (see the file header), so path length is hop count on the filtered
#' graph and expression acts as a tie-break.
#'
#' @param ppi_edges Two-column edge table (`from`, `to`); pass the `ppi_edges` object from 00_setup.R.
#' @param gene_scores Output of [score_genes()].
#' @param tau Detection gate applied to both edge endpoints.
#' @param epsilon Small constant keeping `-log(w + epsilon)` finite (legacy_log only).
#' @param edge_weight_mode One of "legacy_log" (default, reproduces prior runs),
#'   "hop", "bfs", "bounded". See the file header.
#' @param lambda Band width for `edge_weight_mode = "bounded"`; ~0.5 to 1.
#' @return Named list of igraph objects, one per cell type.
build_dynamic_graphs = function(ppi_edges, gene_scores, tau, epsilon,
                                edge_weight_mode = c("legacy_log", "hop", "hop_exact",
                                                     "bfs", "bounded"),
                                lambda = 0.75) {
  edge_weight_mode = match.arg(edge_weight_mode)
  ppi_nodes = unique(c(ppi_edges$from, ppi_edges$to))
  cell_types = sort(unique(gene_scores$cell_type))
  graphs = list()
  for (ct in cell_types) {
    gs = gene_scores[cell_type == ct, .(entrez_id, abund, pct_expr_c)]
    abund_lookup = setNames(gs$abund, gs$entrez_id)
    pct_lookup   = setNames(gs$pct_expr_c, gs$entrez_id)

    e = copy(ppi_edges[, .(from, to)])
    e[, pf := pct_lookup[from]]; e[, pt := pct_lookup[to]]
    e[is.na(pf), pf := 0]; e[is.na(pt), pt := 0]
    e = e[pf >= tau & pt >= tau]                      # edge gate: both endpoints detected

    af = abund_lookup[e$from]; at = abund_lookup[e$to]
    af[is.na(af)] = 0; at[is.na(at)] = 0
    e[, w_edge := af * at]                            # abundance x abundance

    if (edge_weight_mode == "legacy_log") {
      e[, edge_distance := -log(w_edge + epsilon)]
      e[edge_distance < 0, edge_distance := 0]
    } else if (edge_weight_mode %in% c("hop", "hop_exact", "bfs")) {
      e[, edge_distance := 1]
    } else {                                          # "bounded"
      # Detection fractions are already on [0, 1], so the length stays within
      # [1, 1 + lambda]: a hop is still a hop and expression modulates inside a
      # bounded band, instead of epsilon setting every length.
      e[, edge_distance := 1 + lambda * (1 - sqrt(pf * pt))]
    }

    gobj = igraph::graph_from_data_frame(e[, .(from, to, edge_distance)], directed = FALSE,
                                         vertices = data.table(name = ppi_nodes))
    gobj = igraph::simplify(gobj, remove.multiple = TRUE, remove.loops = TRUE,
                            edge.attr.comb = list(edge_distance = "min"))
    graphs[[ct]] = gobj
  }
  graphs
}

#' Per-cell-type node weights for module anchors and drug targets
#'
#' Per-cell-type node weights for the module anchors and the drug targets, from a
#' convex combination of specificity and condition scores, gated by detection.
#' `module_genes` is a plain Entrez vector; its weight is driven purely by the
#' single-cell signal (the old per-gene genetic_score term is removed).
#'
#' @param gene_scores Output of [score_genes()].
#' @param module_genes Character vector of module Entrez IDs.
#' @param dti Drug-target reference table.
#' @param mode_spec Mode spec from [resolve_mode()].
#' @return List with `module_weights` and `target_weights`.
build_node_weights = function(gene_scores, module_genes, dti, mode_spec) {
  lam_m = mode_spec$lambda_module
  lam_t = mode_spec$lambda_target
  cell_types = sort(unique(gene_scores$cell_type))

  gs = gene_scores[, .(entrez_id, cell_type, cell_spec, cond, gate_node, symbol)]
  gs[, gate := as.numeric(gate_node)]
  gs[, combo_m := lam_m[["spec"]] * cell_spec + lam_m[["cond"]] * cond]
  gs[, combo_t := lam_t[["spec"]] * cell_spec + lam_t[["cond"]] * cond]
  gs[, w_t := pmin(pmax(combo_t * gate, 0), 1)]       # target-axis node weight (all genes)

  # module-gene node weights (single-cell signal only)
  module_genes = unique(as.character(module_genes))
  mw = gs[entrez_id %in% module_genes, .(entrez_id, cell_type, combo_m, gate)]
  mw[, w := pmin(combo_m * gate, 1)]
  mw = mw[w > 0]
  module_weights = lapply(split(mw, mw$cell_type), function(d) setNames(d$w, d$entrez_id))

  node_lookup = lapply(split(gs, gs$cell_type), function(d) setNames(d$w_t, d$entrez_id))

  # observed per-drug target weights (cell_type x drug x target)
  dt_t = unique(dti[, .(drugbank_id, drug_name = as.character(drug_name),
                        target_entrez_id = as.character(target_entrez_id),
                        target_symbol = as.character(gene_symbol))])
  by_drug = dt_t[rep(seq_len(.N), length(cell_types))]
  by_drug[, cell_type := rep(cell_types, each = nrow(dt_t))]
  by_drug = merge(by_drug, gs[, .(entrez_id, cell_type, w_t)],
                  by.x = c("target_entrez_id", "cell_type"),
                  by.y = c("entrez_id", "cell_type"), all.x = TRUE)
  by_drug[is.na(w_t), w_t := 0]
  setnames(by_drug, "w_t", "weight")
  by_drug[, active := weight > 0]

  list(module_weights = module_weights,
       target_weights = list(by_drug = by_drug, node_lookup = node_lookup))
}

#' Context-specific (dynamic) proximity per drug and cell type
#'
#' Dynamic proximity for every drug in every cell type: observed weighted
#' distance to the module, matched permutation null(s), z, empirical p, FDR.
#'
#' The null is matched on degree, abundance and (when several cell types exist)
#' specificity; a second null additionally matched on the condition score is run
#' when a reference condition was supplied.
#'
#' @param dynamic_graphs Output of [build_dynamic_graphs()].
#' @param module_weights,target_weights Elements of [build_node_weights()].
#' @param gene_scores Output of [score_genes()].
#' @param mode Mode spec from [resolve_mode()].
#' The empirical p uses the conservative `(1 + #{null <= observed}) / (B + 1)`
#' convention (Phipson & Smyth), matching [run_static_proximity()]. The older
#' `#{null <= observed} / B` form returns p = 0 for a drug closer than every
#' null draw, which inflates the supported set and gives BH nothing to work with.
#'
#' @param n_perm Number of matched permutations. At n_perm = 50 the smallest
#'   attainable p is 1/51 = 0.0196, which is too coarse for BH over thousands of
#'   drug x cell-type rows and cannot pass the Sidak adjustment in
#'   [build_drug_contract()] at all (with K = 8 cell types the floor is 0.146).
#'   Use >= 500 for a final run; 1000 if `edge_weight_mode = "bfs"` is adopted,
#'   since that makes each permutation several times cheaper.
#' @param seed RNG seed.
#' @param epsilon Small constant for the weighted-distance transform.
#' @param edge_weight_mode Must match the mode used to build `dynamic_graphs`.
#' @param anchor_scale How much of one hop the module-anchor penalty spans, for
#'   `edge_weight_mode` "hop" and "bounded". NULL takes the mode default (0.99
#'   and 0.5). Pass 0.5 with "hop" to preserve the legacy ranking; see the note
#'   in [calc_module_distance_vector()].
#' @return data.table with one row per drug x cell type; carries a `status`
#'   attribute recording per-cell-type outcomes.
run_dynamic_proximity = function(dynamic_graphs, module_weights, target_weights, gene_scores,
                                 mode, n_perm = 500, seed = 1, epsilon = 1e-6,
                                 edge_weight_mode = c("legacy_log", "hop", "hop_exact",
                                                      "bfs", "bounded"),
                                 anchor_scale = NULL) {
  edge_weight_mode = match.arg(edge_weight_mode)
  if (n_perm < 500)
    warning("n_perm = ", n_perm, ": smallest attainable p is ", signif(1 / (n_perm + 1), 3),
            ". Use n_perm >= 500 for a final run (see ?run_dynamic_proximity).", call. = FALSE)
  set.seed(seed)
  by_drug    = target_weights$by_drug
  node_lookup = target_weights$node_lookup
  cell_types = names(dynamic_graphs)
  multi_ct      = isTRUE(mode$axes_active$spec)
  run_cond_null = isTRUE(mode$run_cond_null)

  main_cols = c("degree_bin", "abundance_bin")
  if (multi_ct) main_cols = c(main_cols, "specificity_bin")
  null_defs = list(main = main_cols)
  if (run_cond_null) null_defs[["cond"]] = c(main_cols, "condition_bin")

  result_rows = list(); status_rows = list()

  for (ct in cell_types) {
    g = dynamic_graphs[[ct]]
    nodes = igraph::V(g)$name
    mw = module_weights[[ct]]
    if (!is.null(mw)) mw = mw[names(mw) %in% nodes]   # permutation needs graph-node anchors
    if (is.null(mw) || length(mw) < 2) {
      status_rows[[length(status_rows) + 1]] =
        data.table(cell_type = ct, status = "skipped_module_lt2", n_module = length(mw)); next
    }

    module_dist = calc_module_distance_vector(g, mw, nodes, epsilon,
                                              edge_weight_mode = edge_weight_mode,
                                              anchor_scale = anchor_scale)

    gs = gene_scores[cell_type == ct]
    bin = data.table(entrez_id = nodes, degree = as.numeric(igraph::degree(g, v = nodes)))
    bin = merge(bin, gs[, .(entrez_id, abund, cell_spec, cond)], by = "entrez_id", all.x = TRUE)
    for (cc in c("abund", "cell_spec", "cond")) bin[is.na(get(cc)), (cc) := 0]
    bin[, degree_bin      := make_rank_bins(log10(degree + 1))]
    bin[, abundance_bin   := make_rank_bins(abund)]
    bin[, specificity_bin := make_rank_bins(cell_spec)]
    bin[, condition_bin   := make_rank_bins(cond)]

    tlookup  = node_lookup[[ct]]
    sampling = lapply(null_defs, function(cols) prepare_sampling_index(bin, cols))

    module_gene_ids = names(mw); module_w = as.numeric(mw)

    dct = by_drug[cell_type == ct]
    drug_ids = sort(unique(dct$drugbank_id))

    # --- observed proximity + target-weighted summaries ---------------------
    obs_dist = numeric(); obs_targets = list(); obs_meta = list()
    for (drug in drug_ids) {
      td = dct[drugbank_id == drug]
      w = td$weight; names(w) = td$target_entrez_id
      cand = names(w)[is.finite(w) & w > 0 & names(w) %in% nodes]
      if (!length(cand)) next
      dist_c = module_dist[cand]; wc = w[cand]
      keep = is.finite(dist_c) & is.finite(wc) & wc > 0
      cand = cand[keep]; dist_c = dist_c[keep]; wc = wc[keep]
      if (!length(cand)) next

      gg = gs[match(cand, entrez_id)]
      obs_dist[drug]    = weighted.mean(dist_c, wc)
      obs_targets[[drug]] = cand
      dom_t = cand[order(dist_c)][seq_len(min(5, length(cand)))]
      obs_meta[[drug]] = list(
        drug_name            = td$drug_name[1],
        n_active_targets     = length(cand),
        cell_spec_weighted   = weighted.mean(gg$cell_spec,   wc),
        cond_score_weighted  = weighted.mean(gg$cond,        wc),
        cond_spec_weighted   = weighted.mean(gg$cond_spec,   wc),
        signed_cond_weighted = weighted.mean(gg$signed_cond, wc),
        dominant_targets     = paste(dom_t, collapse = ";")
      )
    }
    obs_drugs = names(obs_dist)
    if (!length(obs_drugs)) {
      status_rows[[length(status_rows) + 1]] =
        data.table(cell_type = ct, status = "no_evaluable_drugs", n_module = length(mw)); next
    }

    dom_module = {
      top = module_gene_ids[order(-module_w)][seq_len(min(5, length(module_gene_ids)))]
      sym = gs$symbol[match(top, gs$entrez_id)]
      paste(sym[!is.na(sym) & sym != ""], collapse = ";")
    }

    acc = lapply(null_defs, function(z) list(
      s  = setNames(numeric(length(obs_drugs)),  obs_drugs),
      ss = setNames(numeric(length(obs_drugs)),  obs_drugs),
      n  = setNames(integer(length(obs_drugs)),  obs_drugs),
      le = setNames(integer(length(obs_drugs)),  obs_drugs)
    ))

    # --- permute module and drug targets together ---------------------------
    for (p in seq_len(n_perm)) {
      for (nm in names(null_defs)) {
        idx = sampling[[nm]]
        r_module = sample_matched_genes(module_gene_ids, idx)
        r_module_dt = data.table(entrez_id = r_module, w = module_w)[, .(w = sum(w)), by = entrez_id]
        r_module_dt = r_module_dt[entrez_id %in% nodes & w > 0]
        if (nrow(r_module_dt) < 2) next
        rmw = setNames(r_module_dt$w, r_module_dt$entrez_id)
        rmod = calc_module_distance_vector(g, rmw, nodes, epsilon,
                                           edge_weight_mode = edge_weight_mode,
                                           anchor_scale = anchor_scale)

        for (drug in obs_drugs) {
          r_tg = sample_matched_genes(obs_targets[[drug]], idx)
          rtw = tlookup[r_tg]; rtw[is.na(rtw)] = 0
          rdist = rmod[r_tg]
          ok = is.finite(rdist) & is.finite(rtw) & rtw > 0
          if (!any(ok)) next
          nd = weighted.mean(rdist[ok], rtw[ok])
          if (!is.finite(nd)) next
          acc[[nm]]$s[drug]  = acc[[nm]]$s[drug]  + nd
          acc[[nm]]$ss[drug] = acc[[nm]]$ss[drug] + nd^2
          acc[[nm]]$n[drug]  = acc[[nm]]$n[drug]  + 1L
          acc[[nm]]$le[drug] = acc[[nm]]$le[drug] + as.integer(nd <= obs_dist[drug])  # empirical p uses <=
        }
      }
    }

    summarize = function(nm) {
      a = acc[[nm]]; n = a$n
      mean = ifelse(n > 0, a$s / n, NA_real_)
      var  = ifelse(n > 1, (a$ss - a$s^2 / n) / (n - 1), NA_real_)
      sd   = sqrt(pmax(var, 0))
      # Conservative empirical p, (1 + #{null <= obs}) / (B + 1). Parity with
      # run_static_proximity(): le/n can return 0, which no permutation test can
      # justify and which BH then cannot separate from a genuine floor.
      list(z = (obs_dist[obs_drugs] - mean) / sd,
           p = ifelse(n > 0, (a$le + 1) / (n + 1), NA_real_),
           null_mean = mean, null_sd = sd, n_null = n)
    }
    na_summary = function() {
      nv = setNames(rep(NA_real_, length(obs_drugs)), obs_drugs)
      list(z = nv, p = nv, null_mean = nv, null_sd = nv,
           n_null = setNames(rep(NA_integer_, length(obs_drugs)), obs_drugs))
    }
    main_s = summarize("main")
    cond_s = if (run_cond_null) summarize("cond") else na_summary()

    for (drug in obs_drugs) {
      m = obs_meta[[drug]]
      result_rows[[length(result_rows) + 1]] = data.table(
        drugbank_id = drug, drug_name = m$drug_name, cell_type = ct,
        observed_distance = obs_dist[drug],
        dyn_z = main_s$z[drug], dyn_p = main_s$p[drug],
        # Reported so build_drug_contract() can flag a degenerate null directly
        # (null_sd == 0 gives z = -Inf, a scale artifact) instead of inferring it
        # from a non-finite z.
        dyn_null_mean = main_s$null_mean[drug], dyn_null_sd = main_s$null_sd[drug],
        dyn_n_null = main_s$n_null[drug],
        dyn_z_condnull = cond_s$z[drug], dyn_p_condnull = cond_s$p[drug],
        cell_spec_weighted   = m$cell_spec_weighted,
        cond_score_weighted  = m$cond_score_weighted,
        cond_spec_weighted   = m$cond_spec_weighted,
        signed_cond_weighted = m$signed_cond_weighted,
        n_active_targets = m$n_active_targets,
        dominant_targets = m$dominant_targets,
        dominant_module_genes = dom_module
      )
    }
    status_rows[[length(status_rows) + 1]] =
      data.table(cell_type = ct, status = "tested", n_module = length(mw))
  }

  res = rbindlist(result_rows, fill = TRUE)
  if (nrow(res)) {
    res[, dyn_fdr := p.adjust(dyn_p, method = "fdr")]
    res[, dyn_fdr_condnull := NA_real_]
    if (any(!is.na(res$dyn_p_condnull)))
      res[!is.na(dyn_p_condnull), dyn_fdr_condnull := p.adjust(dyn_p_condnull, method = "fdr")]
    setorder(res, cell_type, dyn_z)
  }
  attr(res, "status") = rbindlist(status_rows, fill = TRUE)
  res[]
}
