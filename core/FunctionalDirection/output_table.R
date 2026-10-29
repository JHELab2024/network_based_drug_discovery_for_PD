# =============================================================================
# FunctionalDirection : output_table.R
#
# Bulk output assembly for RevERT. Produces ONE self-documenting table per
# module, one row per input drug, so every drug is accounted for (no hasCMap /
# noCMap split):
#
#   drug, drug_name
#   in_perturbation_db   TRUE if the drug has its own signature
#   has_structure        TRUE if it can be fingerprinted (biologics are FALSE)
#   direct_score         RevERT from its own signature, else NA
#   direct_z / direct_p  the same score against the analytic null
#   n_eff, n_overlap     how much evidence the score rests on
#   imputed_score        estimate for a drug with no signature of its own
#   imputation_flag      "safe"/"warn"/"no_consensus"/"single_method_only"
#   trust_tier           A / B / C / NA  (see similarity_imputation.R)
#   nbr1..K_name/sim/score/methods       the neighbours it came from
#
# and, alongside it, a self-contained leave-one-out benchmark: for the drugs that
# DO have a direct score, hide each and predict it from its neighbours, so every
# run reports how reliable the imputation is -- against the majority-sign
# baseline, not against 50%.
#
# n_eff and n_overlap are emitted next to every score deliberately. The overlap
# between a module and the CMap landmark set runs from about 5 to 18 genes, so a
# score can look decisive and rest on almost nothing; N_eff says which.
#
# TWO IMPUTATION PATHS.
#   similarity_sources = NULL (default): structure only, exactly the behaviour
#     this function had before the consensus existed. Nothing changes for an
#     analysis that does not opt in.
#   similarity_sources = list(structure = ..., target = ..., network = ...,
#     class = ...): the multi-similarity consensus. A drug is imputed only when
#     independent definitions of "similar" agree on the reference drugs and those
#     drugs agree on the direction.
#
# The core scoring is per module; module-wise comparison is left to the analysis,
# not done here. Call this once per module (a thin loop in the analysis handles
# several modules).
# =============================================================================

# Depends on direction_score.R, similarity_imputation.R (source them first).


# Build the RevERT table for one module.
#   module_signature    : data.frame(gene, sign, weight)   disease direction for the module
#   perturbation_matrix : genes x drugs matrix of drug effects (columns = drugs WITH a signature,
#                         already resolved to the chosen cellular context)
#   fingerprints        : drugs x bits 0/1 matrix (rownames = drug ids) for drugs WITH a structure
#   drug_names          : named vector drug id -> display name (optional)
#   input_drugs         : the drugs to produce rows for (default: all with a signature or a structure)
#   drug_term           : "sign" or "scaled" (passed to the scoring)
#   top_k               : number of neighbours
#   impute_mode         : "weighted" (similarity-weighted mean, default) or "mean".
#                         Weighted is safer: a Tanimoto-0.9 neighbour then dominates
#                         a Tanimoto-0.2 one instead of counting equally.
#                         Ignored on the consensus path, which fuses by rank.
#   min_overlap         : minimum module-gene overlap for a direct score
#   denominator         : "module" (default) or "measured"; see direction_score_matrix()
#   similarity_sources  : NULL for the structure-only path, or a named list of
#                         similarity matrices for the consensus path
#   consensus           : consensus_defaults(), or an override
#   skeletons           : named vector drug id -> InChIKey skeleton. Used ONLY to
#                         exclude a drug's own salts and stereo forms from the
#                         benchmark, where they would otherwise inflate it.
#   impute_from         : "score" (default) or "z". Averaging neighbour z rather
#                         than raw scores gives an imputed value on a scale that
#                         is comparable across modules, because z already divides
#                         out the module's own N_eff. It changes every imputed
#                         number, so it is an analysis decision, not a default.
# Returns list(table, benchmark, benchmark_summary).
functional_direction_table <- function(module_signature, perturbation_matrix, fingerprints,
                                       drug_names = NULL, input_drugs = NULL,
                                       drug_term = "sign", top_k = 3,
                                       impute_mode = c("weighted", "mean"), min_overlap = 3,
                                       denominator = c("module", "measured"),
                                       similarity_sources = NULL,
                                       consensus = consensus_defaults(),
                                       skeletons = NULL,
                                       impute_from = c("score", "z")) {
  impute_mode <- match.arg(impute_mode)
  denominator <- match.arg(denominator)
  impute_from <- match.arg(impute_from)

  # --- 1. direct score for every drug that has a signature (one matrix multiply) ---
  genes <- module_signature$gene
  w <- if (is.null(module_signature$weight)) rep(1, length(genes)) else module_signature$weight
  S <- matrix(module_signature$sign, ncol = 1, dimnames = list(genes, "M"))
  W <- matrix(w,                      ncol = 1, dimnames = list(genes, "M"))
  direct_mat <- direction_score_matrix(S, W, perturbation_matrix,
                                       drug_term = drug_term, min_overlap = min_overlap,
                                       denominator = denominator)
  direct_scores <- direct_mat["M", ]                      # named by drug (may contain NA)
  direct_z      <- attr(direct_mat, "z")["M", ]
  direct_p      <- attr(direct_mat, "p")["M", ]
  direct_neff   <- attr(direct_mat, "n_eff")["M", ]
  direct_nov    <- attr(direct_mat, "n_overlap")["M", ]
  db_drugs <- colnames(perturbation_matrix)

  if (is.null(input_drugs)) input_drugs <- union(db_drugs, rownames(fingerprints))
  if (is.null(drug_names))  drug_names  <- stats::setNames(input_drugs, input_drugs)

  # what the neighbours are averaged over
  impute_values <- if (impute_from == "z") direct_z else direct_scores

  # neighbour reference pool: drugs that have BOTH a finite direct score and a
  # way of being compared to the query
  use_consensus <- !is.null(similarity_sources) && length(similarity_sources) > 0
  ref_ids <- if (use_consensus) {
    intersect(db_drugs[is.finite(direct_scores)],
              unique(unlist(lapply(similarity_sources, colnames))))
  } else {
    intersect(db_drugs[is.finite(direct_scores)], rownames(fingerprints))
  }
  ref_values <- impute_values[ref_ids]

  # --- 2. one row per input drug ---
  rows <- vector("list", length(input_drugs))
  for (i in seq_along(input_drugs)) {
    d <- input_drugs[i]
    in_db  <- d %in% db_drugs
    struct <- d %in% rownames(fingerprints)
    direct <- if (in_db) direct_scores[[d]] else NA_real_

    nm <- rep(NA_character_, top_k); sim <- rep(NA_real_, top_k)
    sc <- rep(NA_real_, top_k);      mth <- rep(NA_character_, top_k)
    imputed <- NA_real_; flag <- NA_character_
    extra <- NULL

    if (use_consensus) {
      r <- impute_consensus(d, similarity_sources, ref_values, spec = consensus,
                            exclude = d, reference = setdiff(ref_ids, d))
      imputed <- r$row$imputed_score
      flag    <- r$row$imputation_flag
      if (nrow(r$neighbours)) {
        k <- seq_len(min(nrow(r$neighbours), top_k))
        nm[k]  <- drug_names[r$neighbours$neighbour[k]]
        sim[k] <- r$neighbours$similarity[k]          # RRF, not a raw similarity
        sc[k]  <- r$neighbours$score[k]
        mth[k] <- r$neighbours$methods[k]
      }
      extra <- r$row[, setdiff(names(r$row),
                               c("drug", "imputed_score", "imputation_flag")), drop = FALSE]
    } else if (struct) {
      pool <- setdiff(ref_ids, d)                     # neighbours are OTHER reference drugs
      nbr <- find_similar_compounds(fingerprints[d, ], fingerprints[pool, , drop = FALSE],
                                    top_k = top_k)
      if (nrow(nbr)) {
        k <- seq_len(nrow(nbr))
        nm[k]  <- drug_names[nbr$neighbour]
        sim[k] <- nbr$similarity
        sc[k]  <- impute_values[nbr$neighbour]
        mth[k] <- "structure"
        imputed <- if (impute_mode == "weighted" && sum(sim[k]) > 0)
                     sum(sim[k] * sc[k]) / sum(sim[k]) else mean(sc[k])
        if (length(k) >= 2) flag <- assess_consensus(sc[k])$status
      }
    }

    row <- data.frame(drug = d, drug_name = drug_names[[d]],
                      in_perturbation_db = in_db, has_structure = struct,
                      direct_score = direct,
                      direct_z      = if (in_db) direct_z[[d]] else NA_real_,
                      direct_p      = if (in_db) direct_p[[d]] else NA_real_,
                      n_eff         = if (in_db) direct_neff[[d]] else NA_real_,
                      n_overlap     = if (in_db) direct_nov[[d]] else NA_real_,
                      imputed_score = imputed,
                      imputation_flag = flag, stringsAsFactors = FALSE)
    if (!use_consensus) {
      # the single-method path never claims more than tier C: one definition of
      # "similar", and the one measured to have no predictive power alone
      row$imputation_method <- if (is.finite(imputed)) "structure_only" else "none"
      row$trust_tier        <- if (is.finite(imputed)) "C" else NA_character_
    } else {
      row <- cbind(row, extra)
    }
    for (k in seq_len(top_k)) {
      row[[paste0("nbr", k, "_name")]]    <- nm[k]
      row[[paste0("nbr", k, "_sim")]]     <- sim[k]
      row[[paste0("nbr", k, "_score")]]   <- sc[k]
      row[[paste0("nbr", k, "_methods")]] <- mth[k]
    }
    rows[[i]] <- row
  }
  tab <- do.call(rbind, rows)
  rownames(tab) <- NULL

  # --- 3. self-contained leave-one-out benchmark on the reference drugs ---
  if (use_consensus) {
    bench_obj <- benchmark_imputation(ref_values, similarity_sources,
                                      spec = consensus, skeletons = skeletons)
    bench <- bench_obj$per_drug
    benchmark_summary <- bench_obj$summary
  } else {
    bench <- benchmark_imputation(ref_values, fingerprints[ref_ids, , drop = FALSE],
                                  top_k = top_k, mode = impute_mode)
    ok <- bench[is.finite(bench$actual) & is.finite(bench$predicted), , drop = FALSE]
    # The majority-sign baseline belongs next to every reported agreement: if
    # most drugs reverse the module, a constant guess already scores high and a
    # method has to be read against that, not against chance.
    maj <- if (nrow(ok)) sign(sum(sign(ok$actual))) else 0
    if (maj == 0) maj <- 1
    benchmark_summary <- list(
      n_benchmarked  = nrow(ok),
      correlation    = if (nrow(ok) > 2) stats::cor(ok$actual, ok$predicted) else NA_real_,
      sign_agreement = if (nrow(ok) > 0) mean(sign(ok$actual) == sign(ok$predicted)) else NA_real_,
      baseline_majority_sign = if (nrow(ok) > 0) mean(sign(ok$actual) == maj) else NA_real_)
  }

  attr(tab, "impute_from") <- impute_from
  attr(tab, "denominator") <- denominator
  list(table = tab, benchmark = bench, benchmark_summary = benchmark_summary)
}
