# =============================================================================
# FunctionalDirection : hierarchy.R
#
# A drug may have a perturbation signature in several cellular contexts (e.g.
# astrocyte, neural progenitor, a reference cancer line). This resolves the many
# per-context direction scores into one value per drug, following a priority
# order that the CALLER supplies. The priority order is disease specific and is
# therefore an argument, not baked into the model -- for Parkinson's disease the
# analysis passes a CNS-first order (astrocyte -> NPC -> neuron -> ... -> MCF7);
# a different application passes its own.
#
# Two resolution modes:
#   "best_available" : take the score from the highest-priority context that has
#                      one. Simple and interpretable.
#   "consensus"      : average across all available contexts and report whether
#                      they agree in sign. More robust; only possible because we
#                      search a panel of contexts rather than a single cell line
#                      (this is the improvement over single-cell-line pAGE).
# =============================================================================


# Resolve per-context direction scores into one score per drug.
#   score_table : data.frame with columns drug, context, score
#   priority    : ordered character vector of contexts, most preferred first
#   mode        : "best_available" or "consensus"
# Returns one row per drug: the resolved score, which context(s) it came from,
# how many contexts were available, and (for consensus) the sign agreement.
resolve_hierarchy <- function(score_table, priority,
                              mode = c("best_available", "consensus")) {
  mode <- match.arg(mode)
  st <- as.data.frame(score_table, stringsAsFactors = FALSE)
  st <- st[is.finite(st$score) & st$context %in% priority, , drop = FALSE]
  # rank each context by its position in the priority list (1 = most preferred)
  st$rank <- match(st$context, priority)

  drugs <- unique(st$drug)
  out <- vector("list", length(drugs))
  for (i in seq_along(drugs)) {
    rows <- st[st$drug == drugs[i], , drop = FALSE]
    rows <- rows[order(rows$rank), , drop = FALSE]

    if (mode == "best_available") {
      out[[i]] <- data.frame(
        drug = drugs[i], score = rows$score[1], context_used = rows$context[1],
        n_contexts = nrow(rows), sign_agreement = NA_real_,
        stringsAsFactors = FALSE)
    } else {
      signs <- sign(rows$score[rows$score != 0])
      agree <- if (length(signs)) max(mean(signs > 0), mean(signs < 0)) else NA_real_
      out[[i]] <- data.frame(
        drug = drugs[i], score = mean(rows$score),
        context_used = paste(rows$context, collapse = ";"),
        n_contexts = nrow(rows), sign_agreement = agree,
        stringsAsFactors = FALSE)
    }
  }
  res <- do.call(rbind, out)
  rownames(res) <- NULL
  res
}
