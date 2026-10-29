# =============================================================================
# NetworkProximity : static_proximity.R
#
# Static (context-free) network proximity of each drug's targets to a disease
# module on the protein-protein interaction network (Guney / Cheng style):
# the mean shortest-path distance from a drug's targets to the nearest module
# gene, compared with a degree-matched random null.
#
# Symbol-native: graph node names, module genes and drug targets are all gene
# SYMBOLS (whatever ids the graph uses). No Entrez mapping.
#
# Empirical p-value uses the conservative (1 + #{null <= observed}) / (B + 1)
# convention (Phipson & Smyth), which is bounded away from zero. A negative
# z-score means the drug's targets are closer to the module than degree-matched
# chance.
# =============================================================================

suppressPackageStartupMessages({ library(data.table); library(igraph) })


# Draw a degree-matched random gene set: for each real gene, one random gene from
# the same degree bin (sampled without replacement within a bin).
sample_degree_matched <- function(real_genes, degree_table) {
  real_genes <- unique(as.character(real_genes))
  bins <- degree_table[match(real_genes, gene), .(gene, degree_bin)]
  bins <- bins[!is.na(degree_bin)]
  drawn <- character(0)
  for (b in unique(bins$degree_bin)) {
    n <- bins[degree_bin == b, .N]
    pool <- degree_table[degree_bin == b, gene]
    drawn <- c(drawn, sample(pool, n, replace = FALSE))
  }
  unique(drawn)
}


# Shortest distance from EVERY node to the nearest gene in `source_genes`, using
# one breadth-first search from a temporary source node wired to all of them
# (much faster than one search per source gene). Returns a named vector over
# `nodes`; unreachable nodes are NA.
distance_to_gene_set <- function(ppi_graph, source_genes, nodes) {
  source_genes <- intersect(unique(as.character(source_genes)), igraph::V(ppi_graph)$name)
  if (!length(source_genes)) return(stats::setNames(rep(NA_real_, length(nodes)), nodes))

  src <- "__source__"
  while (src %in% igraph::V(ppi_graph)$name) src <- paste0(src, "_")
  g2 <- igraph::add_vertices(ppi_graph, 1, name = src)
  g2 <- igraph::add_edges(g2, as.vector(rbind(src, source_genes)))
  d <- as.numeric(igraph::distances(g2, v = src, to = nodes, weights = NA)) - 1
  names(d) <- nodes
  d[is.infinite(d) | d < 0] <- NA_real_
  d
}


# Mean over a drug's targets of the precomputed distance-to-module, dropping
# unreachable targets. NA if the drug has no reachable target.
mean_target_distance <- function(target_genes, distance_vector) {
  target_genes <- intersect(unique(as.character(target_genes)), names(distance_vector))
  if (!length(target_genes)) return(NA_real_)
  d <- distance_vector[target_genes]
  if (all(is.na(d))) return(NA_real_)
  mean(d, na.rm = TRUE)
}


# Static proximity for every drug.
#   ppi_graph   : igraph PPI (node names = gene symbols)
#   module_genes: character vector of module gene symbols
#   dti         : data.table with columns  drug, drug_name, target_gene
#   n_perm      : number of degree-matched permutations (>= 1000 for publication)
#   seed        : RNG seed
# Returns a data.table keyed by drug with observed distance, z, empirical p, FDR.
run_static_proximity <- function(ppi_graph, module_genes, dti,
                                 n_perm = 1000, seed = 1) {
  set.seed(seed)
  nodes <- igraph::V(ppi_graph)$name
  module_genes <- intersect(unique(as.character(module_genes)), nodes)

  d <- unique(dti[, .(drug, drug_name = as.character(drug_name),
                      target_gene = as.character(target_gene))])
  d <- d[target_gene %in% nodes]
  name_map <- d[, .(drug_name = drug_name[!is.na(drug_name) & drug_name != ""][1]), by = drug]
  drug_targets <- lapply(split(d$target_gene, d$drug), function(x) sort(unique(x)))
  drug_targets <- drug_targets[lengths(drug_targets) > 0]

  degree_table <- data.table(gene = nodes, degree = as.numeric(igraph::degree(ppi_graph)))
  degree_table[, degree_bin := cut(log10(degree + 1),
      breaks = unique(quantile(log10(degree + 1), probs = seq(0, 1, 0.05))),
      include.lowest = TRUE)]

  # observed
  module_dist <- distance_to_gene_set(ppi_graph, module_genes, nodes)
  drug_ids <- names(drug_targets)
  observed <- vapply(drug_ids, function(x) mean_target_distance(drug_targets[[x]], module_dist), numeric(1))
  n_targets <- lengths(drug_targets)
  keep <- !is.na(observed)
  drug_ids <- drug_ids[keep]; observed <- observed[keep]; n_targets <- n_targets[keep]

  # degree-matched null: randomise both the module and each drug's targets
  null <- matrix(NA_real_, nrow = length(drug_ids), ncol = n_perm,
                 dimnames = list(drug_ids, NULL))
  for (p in seq_len(n_perm)) {
    rand_module_dist <- distance_to_gene_set(ppi_graph,
        sample_degree_matched(module_genes, degree_table), nodes)
    for (j in seq_along(drug_ids)) {
      rt <- sample_degree_matched(drug_targets[[drug_ids[j]]], degree_table)
      null[j, p] <- mean_target_distance(rt, rand_module_dist)
    }
  }

  res <- data.table(drug = drug_ids, observed_distance = observed, n_targets = n_targets)
  res[, null_mean := rowMeans(null, na.rm = TRUE)]
  res[, null_sd   := apply(null, 1, sd, na.rm = TRUE)]
  res[, static_z  := (observed_distance - null_mean) / null_sd]
  # rigorous empirical p: (1 + #{null <= observed}) / (B + 1)
  res[, static_p  := vapply(seq_len(.N), function(i)
        (1 + sum(null[i, ] <= observed_distance[i], na.rm = TRUE)) / (n_perm + 1), numeric(1))]
  res[, static_fdr := p.adjust(static_p, method = "fdr")]
  res <- merge(res, name_map, by = "drug", all.x = TRUE)
  setcolorder(res, c("drug", "drug_name"))
  setorder(res, static_z, static_p)
  res[]
}
