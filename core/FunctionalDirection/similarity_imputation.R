# =============================================================================
# FunctionalDirection : similarity_imputation.R
#
# When a drug has no measured perturbation signature, its RevERT score is
# estimated from similar drugs that DO have one ("you are what your neighbours
# are"). The question is what "similar" means, and the answer here is: not one
# thing.
#
# WHY NOT STRUCTURE ALONE. Structural similarity was measured on this data and
# it does not carry the direction: leave-one-out |r| <= 0.09, sign agreement at
# chance, and no recovery even at Tanimoto >= 0.95. Activity cliffs are the
# reason -- two molecules can differ by one atom and act oppositely. So structure
# stays as one voice among several rather than the sole oracle.
#
# THE CONSENSUS RULE. For a query drug a, each similarity method m returns a
# ranked top-K_m list over the reference pool. The consensus neighbours are the
# reference drugs found by at least m_min methods (with two methods that is the
# intersection). The imputed score is the fused-rank-weighted mean of their
# direct scores, and the flag reports how many methods agreed and whether the
# neighbours agree in sign. A drug is imputed only when INDEPENDENT definitions
# of "similar" point at the same reference drugs and those drugs agree.
#
# Two things a plain intersection does not handle, and this does:
#
#   1. Empty intersections. Independent methods overlap rarely -- that is the
#      point -- so many drugs get no consensus. Coverage is controlled by K_m and
#      m_min, both chosen from the leave-one-out benchmark, never by hand, and
#      the fallback is explicit and labelled rather than silent.
#   2. Ties and scale. Target Jaccard and class similarity are coarse and produce
#      many ties. They work as VOTES; the ordering inside the consensus set comes
#      from a reciprocal-rank fusion, not from any one method's raw similarity.
#
# THE CHANNELS. S1 structure, S2 target set, S3 target network, S4 therapeutic
# class. S2 and S3 need nothing new (the DTI and the PPI are already loaded by
# the network component); S4 needs an ATC column.
#
# A NOTE THAT MUST TRAVEL WITH ANY SCREENING-LIBRARY RESULT. Library compounds
# have no known targets and no ATC codes, so S2, S3 and S4 are all unavailable
# and the design collapses to S1 -- precisely the channel measured to have no
# predictive power on its own. Imputation for screened compounds is therefore
# weaker than for annotated drugs BY CONSTRUCTION, not by accident, and it should
# be reported that way rather than left for a reader to discover.
#
# Chemistry (SMILES -> fingerprint bits) is isolated in compute_fingerprints(),
# which uses RDKit through reticulate. Everything else is plain R and can be
# tested with any 0/1 fingerprint matrix or any precomputed similarity matrix, so
# the method does not depend on a chemistry toolkit being installed.
#
# SIMILARITY SOURCE CONTRACT. Every channel is a numeric matrix with rownames =
# query drugs and colnames = reference drugs. A drug absent from a matrix simply
# makes that method UNAVAILABLE for it, and an unavailable method does not count
# toward the number of methods that could have agreed.
# =============================================================================

suppressPackageStartupMessages({ library(Matrix) })

# The target-network channel needs the PPI set-distance helpers, which are
# shared with NetworkProximity/combination.R and bundled beside this file.
# Auto-located when this file is source()d by
# path; otherwise source FunctionalDirection/set_distance.R first, or set
# options(dimer.model_root = ...). Only target_network_similarity() needs them,
# so a missing shared file is a warning here and an error there.
if (!exists(".closest_between_sets", mode = "function")) {
  .fd_find_shared <- function() {
    for (i in seq_len(sys.nframe())) {
      of <- sys.frame(i)$ofile
      if (!is.null(of))
        return(file.path(dirname(normalizePath(of, mustWork = FALSE)), "set_distance.R"))
    }
    root <- getOption("dimer.model_root")
    if (is.null(root)) NULL else file.path(root, "FunctionalDirection", "set_distance.R")
  }
  .fd_shared <- .fd_find_shared()
  if (!is.null(.fd_shared) && file.exists(.fd_shared)) source(.fd_shared)
  rm(.fd_shared, .fd_find_shared)
}


# --- S1 chemistry (the only part that needs RDKit) ---------------------------
# SMILES -> (n_compounds x n_bits) 0/1 matrix of Morgan fingerprints.
# Rows for unparseable SMILES are all zero. Requires the 'cadd' Python env with
# rdkit; pass its interpreter path as `python`.
compute_fingerprints <- function(smiles, n_bits = 2048L, radius = 2L, python = NULL) {
  if (!is.null(python)) reticulate::use_python(python, required = TRUE)
  Chem <- reticulate::import("rdkit.Chem", delay_load = TRUE)

  # Preferred route: the generator API, which is what the screening component uses and
  # what RDKit still supports. GetMorganFingerprintAsBitVect() is deprecated and
  # warns on every call; it stays as a fallback for older RDKit builds.
  gen <- tryCatch({
    rdFP <- reticulate::import("rdkit.Chem.rdFingerprintGenerator", delay_load = FALSE)
    rdFP$GetMorganGenerator(radius = as.integer(radius), fpSize = as.integer(n_bits))
  }, error = function(e) NULL)
  rdMD <- if (is.null(gen))
    reticulate::import("rdkit.Chem.rdMolDescriptors", delay_load = TRUE) else NULL

  bits <- matrix(0L, nrow = length(smiles), ncol = n_bits)
  for (i in seq_along(smiles)) {
    if (is.na(smiles[i]) || smiles[i] == "") next
    mol <- Chem$MolFromSmiles(smiles[i])
    if (is.null(mol)) next
    fp <- if (!is.null(gen)) gen$GetFingerprint(mol)
          else rdMD$GetMorganFingerprintAsBitVect(mol, radius, nBits = n_bits)
    # On-bit extraction goes through ToBitString(), which returns a plain
    # character string of length n_bits and so is stable across RDKit versions.
    # The route via GetOnBits() breaks on RDKit 2026.03: it returns an
    # rdkit.rdBase._vecti that reticulate wraps as an environment, which
    # as.integer() cannot coerce. which() on the bit string is already 1-based.
    on <- which(strsplit(fp$ToBitString(), "", fixed = TRUE)[[1]] == "1")
    bits[i, on] <- 1L
  }
  rownames(bits) <- names(smiles)
  bits
}


# --- Tanimoto similarity (pure R) --------------------------------------------
# Similarity of one query fingerprint to every row of a reference fingerprint
# matrix. For binary fingerprints, Tanimoto = shared / (|q| + |r| - shared).
tanimoto_to_reference <- function(query_bits, reference_matrix) {
  shared    <- as.numeric(reference_matrix %*% query_bits)  # count of shared on-bits
  size_ref  <- rowSums(reference_matrix)
  size_query <- sum(query_bits)
  union     <- size_ref + size_query - shared
  sim <- ifelse(union > 0, shared / union, 0)
  names(sim) <- rownames(reference_matrix)
  sim
}


# --- generic neighbour ranker ------------------------------------------------
# Rank one query's similarities to the reference pool and keep the top_k above a
# floor. Every channel goes through this, so ties break the same way everywhere
# ("min" ranking: tied neighbours share the better rank, which is what makes a
# coarse voting channel like ATC usable in the rank fusion).
#
#   sim_vector     : named numeric, reference id -> similarity
#   top_k          : how many neighbours to keep
#   min_similarity : neighbours below this are dropped
#   exclude        : reference ids to drop first (the query itself, and in the
#                    benchmark its same-skeleton relatives)
# Returns data.frame(neighbour, similarity, rank), best first.
neighbours_from_similarity <- function(sim_vector, top_k = 5, min_similarity = 0,
                                       exclude = character()) {
  sim <- sim_vector[is.finite(sim_vector)]
  if (length(exclude)) sim <- sim[!names(sim) %in% exclude]
  sim <- sim[sim >= min_similarity & sim > 0]
  if (!length(sim))
    return(data.frame(neighbour = character(), similarity = numeric(),
                      rank = integer(), stringsAsFactors = FALSE))
  ord <- order(sim, decreasing = TRUE)
  sim <- sim[ord][seq_len(min(top_k, length(sim)))]
  data.frame(neighbour = names(sim), similarity = as.numeric(sim),
             rank = rank(-as.numeric(sim), ties.method = "min"),
             stringsAsFactors = FALSE)
}


# Find the top_k reference compounds most similar to a query compound.
# Thin wrapper: compute Tanimoto, then rank. Kept for the existing call sites.
find_similar_compounds <- function(query_bits, reference_matrix,
                                   top_k = 3, min_similarity = 0) {
  nb <- neighbours_from_similarity(tanimoto_to_reference(query_bits, reference_matrix),
                                   top_k = top_k, min_similarity = min_similarity)
  nb[, c("neighbour", "similarity"), drop = FALSE]
}


# --- S1: structural similarity as a matrix -----------------------------------
# query x reference Tanimoto. Use when running the consensus; the per-query
# find_similar_compounds() path stays available for structure-only work.
structure_similarity <- function(query_fingerprints, reference_fingerprints = NULL) {
  if (is.null(reference_fingerprints)) reference_fingerprints <- query_fingerprints
  # Filled row by row rather than with t(apply(...)): apply() drops to a vector
  # whenever either side has a single row, and t() then silently returns the
  # transpose of the intended matrix.
  m <- matrix(0, nrow(query_fingerprints), nrow(reference_fingerprints),
              dimnames = list(rownames(query_fingerprints), rownames(reference_fingerprints)))
  for (i in seq_len(nrow(query_fingerprints)))
    m[i, ] <- tanimoto_to_reference(query_fingerprints[i, ], reference_fingerprints)
  m
}


# --- S2: target-set similarity (IDF-weighted Jaccard) ------------------------
# Two drugs are similar when they hit the same proteins. IDF weighting
# (u_t = log(N_drugs / n_drugs_hitting_t)) is what keeps this from being a
# promiscuity detector: a target that half the library hits carries almost no
# information that two drugs share it, while a rare target is strong evidence.
#
#   dti        : data.frame with a drug id column and a target column
#   drug_col, target_col : their names
#   idf        : weight targets by inverse document frequency
#   min_shared : drop pairs with fewer than this many shared targets. The plan's
#                floor is ">= 1 shared target AND J >= 0.20"; the count half is
#                applied here so the J half is a plain similarity floor later.
# Returns a dense drug x drug similarity matrix with a zero diagonal.
target_jaccard <- function(dti, drug_col = "drug", target_col = "target_gene",
                           idf = TRUE, min_shared = 1L) {
  d <- unique(data.frame(drug = as.character(dti[[drug_col]]),
                         target = as.character(dti[[target_col]]),
                         stringsAsFactors = FALSE))
  d <- d[!is.na(d$drug) & d$drug != "" & !is.na(d$target) & d$target != "", ]
  if (!nrow(d)) stop("target_jaccard(): no usable drug-target rows")

  drugs <- sort(unique(d$drug)); targets <- sort(unique(d$target))
  A <- Matrix::sparseMatrix(i = match(d$drug, drugs), j = match(d$target, targets),
                            x = 1, dims = c(length(drugs), length(targets)),
                            dimnames = list(drugs, targets))

  n_hit <- Matrix::colSums(A)
  u <- if (idf) log(length(drugs) / pmax(n_hit, 1)) else rep(1, length(targets))
  u[!is.finite(u) | u < 0] <- 0            # a target every drug hits carries nothing

  Aw <- A %*% Matrix::Diagonal(x = u)
  inter <- as.matrix(Aw %*% Matrix::t(A))  # sum of u over shared targets
  size  <- Matrix::rowSums(Aw)             # weighted set size per drug
  union <- outer(size, size, "+") - inter
  sim <- ifelse(union > 0, inter / union, 0)

  shared_n <- as.matrix(A %*% Matrix::t(A))
  sim[shared_n < min_shared] <- 0
  dimnames(sim) <- list(drugs, drugs)
  diag(sim) <- 0
  attr(sim, "n_shared") <- shared_n
  sim
}


# --- S3: target-network similarity --------------------------------------------
# Drugs whose targets sit close on the interactome without sharing any target:
# same pathway, different protein. sim = exp(-d_ab) with d_ab the Cheng mean
# closest distance between the two target sets, so d_ab = 0 (shared targets)
# gives 1 and d_ab = 1 (adjacent targets) gives exp(-1) = 0.37.
#
# Computed from ONE all-pairs distance matrix over every target, then reduced
# pairwise. The reduction is the identity
#
#     d_ab = ( sum_{t in T_a} m_b[t] + sum_{t in T_b} m_a[t] ) / (|T_a| + |T_b|)
#
# where m_b[t] is t's distance to the nearest member of T_b. That is exactly
# .closest_between_sets() -- the mean of the row minima and the column minima of
# the submatrix -- but it costs one pass per drug instead of one per drug PAIR,
# which is the difference between minutes and days at a few thousand drugs.
#
#   drug_targets : named list, drug id -> character vector of target genes
#   ppi_graph    : igraph PPI (node names are the same gene ids)
# Returns a dense drug x drug similarity matrix with a zero diagonal.
target_network_similarity <- function(drug_targets, ppi_graph) {
  if (!exists(".closest_between_sets", mode = "function"))
    stop("target_network_similarity() needs FunctionalDirection/set_distance.R; ",
         "source() it first or set options(dimer.model_root = '/path/to/model_release/core')")
  nodes <- igraph::V(ppi_graph)$name
  dt <- lapply(drug_targets, function(x) intersect(unique(as.character(x)), nodes))
  dt <- dt[lengths(dt) > 0]
  if (length(dt) < 2) stop("target_network_similarity(): fewer than 2 drugs have PPI targets")

  dmat <- target_distance_matrix(ppi_graph, unlist(dt, use.names = FALSE))
  tg <- rownames(dmat)

  # M[t, b] = distance from target t to the nearest target of drug b
  M <- vapply(dt, function(tb) {
    sub <- dmat[, tb, drop = FALSE]
    suppressWarnings(apply(sub, 1, function(r) { r <- r[is.finite(r)]
                                                 if (!length(r)) NA_real_ else min(r) }))
  }, numeric(length(tg)))
  dimnames(M) <- list(tg, names(dt))

  drugs <- names(dt); n <- length(drugs)
  dab <- matrix(NA_real_, n, n, dimnames = list(drugs, drugs))
  for (i in seq_len(n)) for (j in seq_len(i)) {
    v <- c(M[dt[[i]], j], M[dt[[j]], i])
    v <- v[is.finite(v)]
    dab[i, j] <- dab[j, i] <- if (length(v)) mean(v) else NA_real_
  }
  sim <- exp(-dab)
  sim[!is.finite(sim)] <- 0
  diag(sim) <- 0
  attr(sim, "d_ab") <- dab
  sim
}


# --- S4: therapeutic / mechanism class ---------------------------------------
# Graded ATC agreement: same ATC4 = 1, same ATC3 = 0.67, same ATC2 = 0.33.
# Deliberately coarse and full of ties -- this is a vote about what a drug is
# FOR, not a ranker, and the rank fusion is what turns the vote into an order.
#
#   atc_table : data.frame with a drug id column and an ATC code column; one row
#               per (drug, code), so a drug with several codes gets several rows
# Returns a dense drug x drug similarity matrix with a zero diagonal.
class_similarity <- function(atc_table, drug_col = "drug", atc_col = "atc") {
  a <- unique(data.frame(drug = as.character(atc_table[[drug_col]]),
                         atc  = toupper(trimws(as.character(atc_table[[atc_col]]))),
                         stringsAsFactors = FALSE))
  a <- a[!is.na(a$drug) & a$drug != "" & !is.na(a$atc) & nchar(a$atc) >= 3, ]
  if (!nrow(a)) stop("class_similarity(): no usable ATC rows")
  drugs <- sort(unique(a$drug))

  level_match <- function(width) {
    code <- substr(a$atc, 1, width)
    keep <- nchar(code) == width
    if (!any(keep)) return(matrix(0, length(drugs), length(drugs),
                                  dimnames = list(drugs, drugs)))
    codes <- sort(unique(code[keep]))
    A <- Matrix::sparseMatrix(i = match(a$drug[keep], drugs), j = match(code[keep], codes),
                              x = 1, dims = c(length(drugs), length(codes)))
    m <- as.matrix(A %*% Matrix::t(A))
    dimnames(m) <- list(drugs, drugs)
    (m > 0) * 1
  }

  sim <- pmax(1.00 * level_match(5),      # ATC4, e.g. N04BA
              0.67 * level_match(4),      # ATC3, e.g. N04B
              0.33 * level_match(3))      # ATC2, e.g. N04
  dimnames(sim) <- list(drugs, drugs)
  diag(sim) <- 0
  sim
}


# --- the consensus rule -------------------------------------------------------
# Fuse per-method neighbour lists into one ordered candidate set.
#
#   lists : named list of data.frame(neighbour, similarity, rank), one per
#           AVAILABLE method (an unavailable method must be absent, not empty)
#   m_min : how many methods must list a neighbour for it to be a candidate.
#           With M = 2 this is the intersection. With M >= 3 a strict
#           intersection of all lists is almost always empty, so ">= 2" keeps the
#           selectivity the intersection was meant to provide while letting the
#           third and fourth methods add coverage instead of destroying it.
#   k     : how many candidates to keep
#   c     : reciprocal-rank-fusion constant, RRF(r) = sum_m 1 / (c + rank_m(r)).
#           The textbook 60 is for long lists; with K = 5 it flattens the
#           ordering to nothing, so use a small c and let the benchmark confirm it.
# Returns data.frame(neighbour, rrf, n_methods, methods), best first.
consensus_neighbours <- function(lists, m_min = 2L, k = 3L, c = 10) {
  lists <- lists[vapply(lists, function(x) is.data.frame(x) && nrow(x) > 0, logical(1))]
  empty <- data.frame(neighbour = character(), rrf = numeric(),
                      n_methods = integer(), methods = character(),
                      stringsAsFactors = FALSE)
  if (length(lists) < m_min) return(empty)

  long <- do.call(rbind, lapply(names(lists), function(m)
    data.frame(neighbour = lists[[m]]$neighbour, rank = lists[[m]]$rank,
               method = m, stringsAsFactors = FALSE)))
  long$contrib <- 1 / (c + long$rank)

  by_nb <- split(long, long$neighbour)
  out <- do.call(rbind, lapply(names(by_nb), function(nb) {
    x <- by_nb[[nb]]
    ms <- sort(unique(x$method))
    data.frame(neighbour = nb, rrf = sum(x$contrib), n_methods = length(ms),
               methods = paste(ms, collapse = ";"), stringsAsFactors = FALSE)
  }))

  out <- out[out$n_methods >= m_min, , drop = FALSE]
  if (!nrow(out)) return(empty)
  out <- out[order(-out$rrf, out$neighbour), c("neighbour", "rrf", "n_methods", "methods")]
  rownames(out) <- NULL
  out[seq_len(min(k, nrow(out))), , drop = FALSE]
}


# Check whether the neighbours agree well enough to trust an imputed value.
#   neighbour_scores     : the direction scores of the neighbours (already known)
#   neighbour_signatures : optional genes x neighbours matrix of their perturbation
#                          profiles, used to report signature-level agreement
#   score_tolerance      : max spread (max - min) of neighbour scores for "safe"
#   min_signature_cor    : min mean pairwise correlation of neighbour signatures
#   require_sign_agreement : neighbours that disagree about the DIRECTION cannot
#                          be "safe" whatever their spread. A pair at +0.2 and
#                          -0.2 is inside a 0.5 tolerance but says the drug both
#                          reverses and reinforces the module, which is not a
#                          usable prediction.
# Returns a list: status ("safe"/"warn"), score spread, signature correlation.
assess_consensus <- function(neighbour_scores, neighbour_signatures = NULL,
                             score_tolerance = 0.5, min_signature_cor = 0.3,
                             require_sign_agreement = TRUE) {
  s <- neighbour_scores[is.finite(neighbour_scores)]
  score_spread <- if (length(s) >= 2) max(s) - min(s) else NA_real_

  nz <- sign(s[s != 0])
  sign_agree <- if (!length(nz)) NA else length(unique(nz)) == 1

  signature_cor <- NA_real_
  if (!is.null(neighbour_signatures) && ncol(neighbour_signatures) >= 2) {
    cm <- suppressWarnings(stats::cor(neighbour_signatures, use = "pairwise.complete.obs"))
    signature_cor <- mean(cm[lower.tri(cm)], na.rm = TRUE)
  }

  safe <- isTRUE(score_spread <= score_tolerance) &&
          (is.na(signature_cor) || signature_cor >= min_signature_cor) &&
          (!require_sign_agreement || is.na(sign_agree) || isTRUE(sign_agree))
  list(status = if (safe) "safe" else "warn",
       score_spread = score_spread, signature_cor = signature_cor,
       sign_agreement = sign_agree, n_neighbours = length(s))
}


# Impute a direction score for a drug with no signature, from its neighbours.
#   neighbours     : data.frame(neighbour, similarity) from find_similar_compounds(),
#                    or a consensus set whose `similarity` column carries the RRF
#   known_scores   : named vector, neighbour id -> its (known) direction score
#   mode           : "mean"     -> simple average of the top-k neighbour scores
#                    "weighted" -> similarity-weighted average
#                    "individual" -> return each neighbour separately, no averaging
# Returns a list with the imputed score(s) and the neighbours used.
impute_direction_score <- function(neighbours, known_scores,
                                   mode = c("mean", "weighted", "individual")) {
  mode <- match.arg(mode)
  ids  <- neighbours$neighbour
  sc   <- known_scores[ids]
  sim  <- neighbours$similarity
  keep <- is.finite(sc)
  ids  <- ids[keep]; sc <- sc[keep]; sim <- sim[keep]

  if (!length(sc)) {
    return(list(score = NA_real_, mode = mode,
                neighbours = data.frame(neighbour = character(),
                                        similarity = numeric(), score = numeric())))
  }
  nb <- data.frame(neighbour = ids, similarity = sim, score = as.numeric(sc),
                   stringsAsFactors = FALSE)
  score <- switch(mode,
                  mean       = mean(sc),
                  weighted   = if (sum(sim) > 0) sum(sim * sc) / sum(sim) else mean(sc),
                  individual = NA_real_)   # caller reads per-neighbour scores
  list(score = score, mode = mode, neighbours = nb)
}


# --- putting it together ------------------------------------------------------
# Default consensus settings. Every number here is a STARTING POINT to be
# replaced by the benchmark, not a tuned value: the floors come from where sign
# agreement falls to the majority-class baseline, and K/m_min from the coverage
# you are willing to accept.
consensus_defaults <- function() {
  list(m_min = 2L, K = 5L, k = 3L, c = 10,
       floors = list(structure = 0.40,   # Tanimoto
                     target    = 0.20,   # IDF-weighted Jaccard
                     network   = exp(-1),# d_ab <= 1: shared or adjacent targets
                     class     = 0.67))  # ATC3 or better
}

# Per-method neighbour lists for one query, over the available methods only.
#
# `reference` restricts the candidate pool, and passing it matters: the top-K is
# taken before anything checks whether a neighbour HAS a known score, so without
# the restriction a query can spend all K slots on drugs that cannot contribute
# and come back empty for no visible reason.
similarity_lists <- function(query, similarity_sources, spec = consensus_defaults(),
                             exclude = character(), reference = NULL) {
  out <- list()
  for (m in names(similarity_sources)) {
    S <- similarity_sources[[m]]
    if (!(query %in% rownames(S))) next          # method unavailable for this query
    cols <- colnames(S)
    if (!is.null(reference)) cols <- intersect(cols, reference)
    if (!length(cols)) next
    floor_m <- spec$floors[[m]]
    if (is.null(floor_m)) floor_m <- 0
    nb <- neighbours_from_similarity(stats::setNames(as.numeric(S[query, cols]), cols),
                                     top_k = spec$K, min_similarity = floor_m,
                                     exclude = unique(c(query, exclude)))
    if (nrow(nb)) out[[m]] <- nb
  }
  out
}

# Full imputation for one query: per-method imputations, the consensus, the
# flags, and the trust tier. Returns a one-row data.frame plus the neighbour set.
#
# THE FALLBACK IS EXPLICIT AND NEVER SILENT. When there is no consensus,
# `imputed_score` is NA with a flag saying why, and the per-method `imp_*`
# columns are still filled so an analysis can adopt the benchmark-best single
# method under a lower trust label if it chooses to. What it must not do is
# quietly become the primary number.
#
# Trust tiers:
#   A  consensus neighbours, flag "safe", every available single-method
#      imputation agrees in sign with the consensus
#   B  consensus neighbours, but "warn" or the single methods disagree in sign
#   C  no consensus; a single-method imputation exists
#   NA nothing imputable
impute_consensus <- function(query, similarity_sources, known_scores,
                             spec = consensus_defaults(), exclude = character(),
                             reference = NULL) {
  if (is.null(reference)) reference <- names(known_scores)[is.finite(known_scores)]
  lists <- similarity_lists(query, similarity_sources, spec, exclude = exclude,
                            reference = reference)
  m_avail <- names(lists)

  per_method <- stats::setNames(rep(NA_real_, length(similarity_sources)),
                                paste0("imp_", names(similarity_sources)))
  for (m in m_avail) {
    nb <- lists[[m]][seq_len(min(spec$k, nrow(lists[[m]]))), , drop = FALSE]
    per_method[[paste0("imp_", m)]] <- impute_direction_score(nb, known_scores,
                                                              mode = "weighted")$score
  }

  cons <- if (length(m_avail) >= spec$m_min)
    consensus_neighbours(lists, m_min = spec$m_min, k = spec$k, c = spec$c) else
    consensus_neighbours(list())

  flag <- NA_character_; imputed <- NA_real_; status <- NA_character_
  nbr <- data.frame(neighbour = character(), similarity = numeric(),
                    score = numeric(), methods = character(), stringsAsFactors = FALSE)

  if (length(m_avail) < spec$m_min) {
    flag <- if (length(m_avail)) "single_method_only" else "no_similarity"
  } else if (!nrow(cons)) {
    flag <- "no_consensus"
  } else {
    imp <- impute_direction_score(
      data.frame(neighbour = cons$neighbour, similarity = cons$rrf,
                 stringsAsFactors = FALSE), known_scores, mode = "weighted")
    imputed <- imp$score
    ac <- assess_consensus(imp$neighbours$score, require_sign_agreement = TRUE)
    status <- ac$status; flag <- ac$status
    nbr <- merge(imp$neighbours, cons[, c("neighbour", "n_methods", "methods")],
                 by = "neighbour", all.x = TRUE)
    nbr <- nbr[order(-nbr$similarity), , drop = FALSE]
  }

  single <- per_method[is.finite(per_method)]
  n_sign_agree <- if (is.finite(imputed) && length(single))
    sum(sign(single) == sign(imputed)) else NA_integer_

  tier <- if (!nrow(nbr)) {
    if (length(single)) "C" else NA_character_
  } else if (identical(status, "safe") &&
             (!length(single) || all(sign(single) == sign(imputed)))) "A" else "B"

  row <- data.frame(
    drug = query,
    imputed_score = imputed,
    imputation_flag = flag,
    imputation_method = if (nrow(nbr)) "consensus" else "none",
    consensus_level = if (nrow(nbr)) nbr$n_methods[1] else NA_integer_,
    n_consensus_neighbours = nrow(nbr),
    consensus_methods = if (nrow(nbr)) paste(sort(unique(unlist(strsplit(nbr$methods, ";")))),
                                             collapse = ";") else NA_character_,
    n_methods_available = length(m_avail),
    n_methods_sign_agree = n_sign_agree,
    trust_tier = tier,
    stringsAsFactors = FALSE)
  row <- cbind(row, as.data.frame(as.list(per_method), stringsAsFactors = FALSE,
                                  check.names = FALSE))
  list(row = row, neighbours = nbr, lists = lists, consensus = cons)
}


# --- leave-one-out benchmark --------------------------------------------------
# For drugs that DO have a known score, hide each one, predict it from its
# neighbours, and compare. This is where K_m, m_min, c and the per-method floors
# come from; none of them should be set by hand.
#
#   known_scores       : named vector, drug id -> known direction score
#   similarity_sources : named list of similarity matrices (the contract at the
#                        top of this file), OR a bare fingerprint matrix for the
#                        structure-only case, which keeps the old call working
#   spec               : consensus_defaults(), or an override
#   skeletons          : optional named vector, drug id -> InChIKey skeleton.
#                        Same-skeleton reference drugs (salts, stereo forms) are
#                        legitimate neighbours in production but must be excluded
#                        HERE or the benchmark measures the pipeline's ability to
#                        find a drug's own hydrochloride and reports it as skill.
#   top_k, mode        : legacy arguments for the structure-only path
#
# Returns a list(per_drug, summary). `summary` carries one row per method plus
# the consensus AND the majority-sign baseline: if most drugs reverse a module, a
# constant guess already scores well, and a method must be read against that, not
# against 50%.
benchmark_imputation <- function(known_scores, similarity_sources,
                                 spec = consensus_defaults(), skeletons = NULL,
                                 top_k = 3, mode = "mean") {

  # legacy structure-only call: benchmark_imputation(scores, fingerprint_matrix)
  legacy <- !is.list(similarity_sources)
  if (legacy) {
    ref <- similarity_sources
    ids <- intersect(names(known_scores), rownames(ref))
    out <- vector("list", length(ids))
    for (i in seq_along(ids)) {
      id <- ids[i]; others <- setdiff(ids, id)
      nb <- find_similar_compounds(ref[id, ], ref[others, , drop = FALSE], top_k = top_k)
      imp <- impute_direction_score(nb, known_scores[others], mode = mode)
      out[[i]] <- data.frame(drug = id, actual = known_scores[id], predicted = imp$score,
                             top_similarity = if (nrow(nb)) nb$similarity[1] else NA_real_,
                             n_neighbours = nrow(nb), stringsAsFactors = FALSE)
    }
    res <- do.call(rbind, out); rownames(res) <- NULL
    return(res)
  }

  # A drug needs a known score and a presence in at least ONE source. Requiring
  # it in every source would silently restrict the benchmark to the drugs with
  # complete annotation, which are exactly the drugs the consensus finds easiest.
  ids <- intersect(names(known_scores),
                   unique(unlist(lapply(similarity_sources, rownames))))
  ids <- ids[is.finite(known_scores[ids])]
  methods <- names(similarity_sources)

  rows <- vector("list", length(ids))
  for (i in seq_along(ids)) {
    id <- ids[i]
    drop <- if (!is.null(skeletons) && !is.na(skeletons[id]))
      names(skeletons)[skeletons == skeletons[[id]]] else character()
    r <- impute_consensus(id, similarity_sources, known_scores[setdiff(ids, id)],
                          spec = spec, exclude = unique(c(id, drop)),
                          reference = setdiff(ids, unique(c(id, drop))))
    r$row$actual <- as.numeric(known_scores[[id]])
    r$row$top_rrf <- if (nrow(r$neighbours)) r$neighbours$similarity[1] else NA_real_
    rows[[i]] <- r$row
  }
  per_drug <- do.call(rbind, rows)
  if (is.null(per_drug)) return(list(per_drug = NULL, summary = NULL))
  rownames(per_drug) <- NULL

  score_one <- function(pred, label) {
    ok <- is.finite(pred) & is.finite(per_drug$actual)
    data.frame(method = label, n = sum(ok),
               coverage = if (nrow(per_drug)) sum(ok) / nrow(per_drug) else NA_real_,
               sign_agreement = if (any(ok)) mean(sign(pred[ok]) == sign(per_drug$actual[ok])) else NA_real_,
               spearman = if (sum(ok) > 2)
                 suppressWarnings(stats::cor(pred[ok], per_drug$actual[ok], method = "spearman")) else NA_real_,
               stringsAsFactors = FALSE)
  }

  summ <- rbind(
    score_one(per_drug$imputed_score, "consensus"),
    do.call(rbind, lapply(methods, function(m) score_one(per_drug[[paste0("imp_", m)]], m))))

  # sign agreement within the "safe" flag, which is the number that justifies the flag
  safe <- per_drug$imputation_flag %in% "safe" & is.finite(per_drug$imputed_score)
  summ <- rbind(summ, data.frame(
    method = "consensus_safe_only", n = sum(safe),
    coverage = if (nrow(per_drug)) sum(safe) / nrow(per_drug) else NA_real_,
    sign_agreement = if (any(safe))
      mean(sign(per_drug$imputed_score[safe]) == sign(per_drug$actual[safe])) else NA_real_,
    spearman = if (sum(safe) > 2)
      suppressWarnings(stats::cor(per_drug$imputed_score[safe], per_drug$actual[safe],
                                  method = "spearman")) else NA_real_,
    stringsAsFactors = FALSE))

  # the baseline every method must beat: guess the module's majority sign
  maj <- if (any(is.finite(per_drug$actual)))
    sign(sum(sign(per_drug$actual[is.finite(per_drug$actual)]))) else 0
  if (maj == 0) maj <- 1
  summ <- rbind(summ, data.frame(
    method = "baseline_majority_sign", n = sum(is.finite(per_drug$actual)),
    coverage = 1,
    sign_agreement = mean(sign(per_drug$actual[is.finite(per_drug$actual)]) == maj),
    spearman = NA_real_, stringsAsFactors = FALSE))

  list(per_drug = per_drug, summary = summ)
}
