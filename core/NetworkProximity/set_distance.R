# =============================================================================
# NetworkProximity : set_distance.R
#
# Distances between two sets of genes on the PPI. Both components need this and
# neither should have to source the other, so it lives here.
#
#   NetworkProximity/combination.R  uses it for the Cheng separation
#                                      s_ab = d_ab - (d_aa + d_bb) / 2
#   FunctionalDirection/similarity_imputation.R  uses it for the target-network
#                                      similarity channel exp(-d_ab), which finds
#                                      same-pathway drugs that share no target
#
# The mean-closest convention is Cheng's: for each member of a set take the
# distance to its nearest member of the other set, then average over both
# directions so the measure is symmetric. A lone target has no internal spread,
# so the within-set distance of a single-target drug is 0 by convention.
#
# Unreachable pairs are NA rather than Inf throughout, and every aggregation
# skips them: on a disconnected interactome an Inf would otherwise propagate and
# turn one unreachable target into a missing drug.
# =============================================================================

# Minimum / mean over the finite entries; NA when nothing is finite.
.safe_min  = function(x) { x = x[is.finite(x)]; if (!length(x)) NA_real_ else min(x) }
.safe_mean = function(x) { x = x[is.finite(x)]; if (!length(x)) NA_real_ else mean(x) }

# Mean closest distance within one target set (0 for a single target, by the
# Cheng convention: a lone target has no internal spread).
.closest_within_set = function(one_set, dmat) {
  one_set = intersect(unique(as.character(one_set)), rownames(dmat))
  if (length(one_set) == 0) return(NA_real_)
  if (length(one_set) == 1) return(0)
  d = dmat[one_set, one_set, drop = FALSE]
  diag(d) = NA_real_
  .safe_mean(apply(d, 1, .safe_min))
}

# Mean closest distance between two target sets, symmetrised over both directions.
.closest_between_sets = function(set_a, set_b, dmat) {
  set_a = intersect(unique(as.character(set_a)), rownames(dmat))
  set_b = intersect(unique(as.character(set_b)), rownames(dmat))
  if (length(set_a) == 0 || length(set_b) == 0) return(NA_real_)
  d = dmat[set_a, set_b, drop = FALSE]
  .safe_mean(c(apply(d, 1, .safe_min), apply(d, 2, .safe_min)))
}

# One all-pairs shortest-path matrix over every gene that will ever be looked up.
# Computed once and reused: this is by far the most expensive step, and doing it
# per drug pair is what makes a naive combination screen intractable.
target_distance_matrix = function(ppi_graph, genes) {
  stopifnot("pass an igraph ppi_graph" = inherits(ppi_graph, "igraph"))
  genes = sort(unique(as.character(genes)))
  genes = genes[genes %in% igraph::V(ppi_graph)$name]
  if (!length(genes)) stop("none of the supplied genes are nodes of the PPI")
  dmat = igraph::distances(ppi_graph, v = genes, to = genes)
  dimnames(dmat) = list(genes, genes)
  dmat[is.infinite(dmat)] = NA_real_
  dmat
}
