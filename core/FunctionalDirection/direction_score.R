# =============================================================================
# FunctionalDirection : direction_score.R
#
# RevERT -- Reversal of Expression Relative to the disease Transcriptome.
#
# The core scoring math. Given (i) a disease direction signature for a set of
# module genes and (ii) a drug's transcriptional effect on those genes, this
# asks whether the drug pushes the module genes OPPOSITE to the disease
# direction (reversal, a candidate cure) or in the SAME direction (reinforcement,
# makes it worse).
#
# This is a weighted generalisation of the pAGE / Connectivity-Map reversal idea.
# pAGE uses equal weights and the sign of the drug effect; here each gene can be
# weighted (e.g. by how strongly it is dysregulated in disease, or by curated
# gene-disease confidence), and the drug term can be a sign or a scaled magnitude.
#
# For one drug d and one module M with overlapping genes g in Lambda:
#
#     s_g  = sign(disease log fold change)      in {-1, +1}   (up / down in disease)
#     d_g  = drug effect term                                 ("sign" or "scaled")
#     w_g  = gene weight                          >= 0          (default 1)
#
#     RevERT = - sum_g ( w_g * s_g * d_g ) / sum_g ( w_g )     in [-1, +1]
#
#   score > 0 : the drug reverses the disease direction on this module (good)
#   score < 0 : the drug reinforces the disease direction (bad)
#   score = 0 : no net directional relationship
#
# --- the analytic null -------------------------------------------------------
#
# Under the null that the drug signature is independent of the disease direction,
# with E[d_g] = 0,
#
#     E[RevERT]   = 0
#     Var(RevERT) = sum_g w_g^2 Var(d_g) / ( sum_g w_g )^2
#
# For the sign term Var(d_g) = 1 over the measured genes, so
#
#     sigma = sqrt( sum w^2 ) / sum w  =  1 / sqrt(N_eff),
#     N_eff = ( sum w )^2 / sum w^2
#
# Two things fall out of that, and both are reported here:
#
#   1. pAGE is the equal-weight, sign-term special case. Setting every w_g = 1
#      gives sigma = 1/sqrt(N), which is the published result. "More general" is
#      then a reduction rather than a claim.
#   2. N_eff is the honest sample size. A weighted score behaves like an
#      unweighted score on N_eff genes, which says directly how much evidence a
#      value rests on. The module-by-CMap-landmark overlap runs from 5 to 18
#      genes, so this number belongs next to every reported score.
#
# For the "scaled" term Var(d_g) != 1 and is estimated from the drug's own
# signature (the same tanh scale that entered the score, applied to every gene
# the drug measured -- far more genes than the module overlap, so the estimate is
# not itself noise-dominated). The same formula then applies.
#
# An analytic null also removes a permutation cost, which is why the directional
# layer adds none to the dynamic component.
#
# TWO ASSUMPTIONS THE NULL RESTS ON, BOTH REPORTED SO THEY CAN BE CHECKED.
#
#   E[d_g] = 0 requires the drug signature to carry no global shift. A signature
#   that is mostly up-regulation has E[d_g] > 0, and the score then has a nonzero
#   mean under the null that sigma does not capture. `drug_term_mean` reports that
#   shift per drug; `center_drug_term = TRUE` removes it. It is off by default
#   because centering changes every score, so it is an analysis decision.
#
#   The normal approximation is marginal at N_eff of 5 to 18, which is the range
#   the module-by-landmark overlap actually delivers. `max_weight_share` reports
#   whether one gene carries most of the weight, which is when the approximation
#   is worst. For the sign term the exact null is the Rademacher distribution of
#   sum_g w_g eps_g and is enumerable at these sizes; that belongs with the
#   Supplementary Note's derivation and is not done here.
#
# Nothing here is disease specific. All inputs are named vectors / tables keyed
# by gene symbol.
# =============================================================================


# Robust spread used by the "scaled" drug term. Split out from
# scale_drug_effect() so the null can apply the SAME scale to the drug's whole
# signature without changing the score.
.robust_scale <- function(effect) {
  finite <- effect[is.finite(effect)]
  scale <- stats::mad(finite)
  if (!is.finite(scale) || scale <= 0) scale <- stats::sd(finite)
  if (!is.finite(scale) || scale <= 0) scale <- 1
  scale
}


# Squash a drug effect (e.g. a CMap z-score) into (-1, +1) for the "scaled" term.
# scale is a robust spread; tanh keeps large effects bounded so one extreme gene
# cannot dominate the score.
scale_drug_effect <- function(effect, scale = NULL) {
  if (is.null(scale)) scale <- .robust_scale(effect)
  tanh(effect / scale)
}


# Score one drug against one module.
#
#   disease_signature : data.frame with columns
#                         gene    (symbol)
#                         sign    (-1 or +1 ; direction of change in disease)
#                         weight  (optional, >= 0 ; defaults to 1 for every gene)
#   drug_effect       : named numeric vector, gene symbol -> drug effect
#                       (e.g. CMap perturbation z-score). Genes not measured by
#                       the drug are simply absent.
#   drug_term         : "sign"   -> d_g = sign(drug effect)          (pAGE style)
#                       "scaled" -> d_g = tanh(drug effect / spread) (magnitude aware)
#   min_overlap       : minimum number of overlapping genes; below this the score
#                       is returned as NA (too little evidence).
#
# Returns a one-row data.frame with the score, its supporting counts, and the
# analytic null: n_eff, null_sd, z and a two-sided p.
direction_score <- function(disease_signature, drug_effect,
                            drug_term = c("sign", "scaled"),
                            min_overlap = 3, center_drug_term = FALSE) {
  drug_term <- match.arg(drug_term)

  sig <- as.data.frame(disease_signature)
  if (is.null(sig$weight)) sig$weight <- 1
  # keep genes with a real disease direction and a real weight
  sig <- sig[is.finite(sig$sign) & sig$sign != 0 &
               is.finite(sig$weight) & sig$weight >= 0, , drop = FALSE]

  # overlap: genes that are in the signature and measured by the drug
  genes <- intersect(sig$gene, names(drug_effect))
  genes <- genes[is.finite(drug_effect[genes]) & drug_effect[genes] != 0]

  n_overlap <- length(genes)
  if (n_overlap < min_overlap) {
    return(data.frame(score = NA_real_, n_overlap = n_overlap,
                      reversal_n = NA_integer_, reinforce_n = NA_integer_,
                      weight_sum = NA_real_, n_eff = NA_real_,
                      max_weight_share = NA_real_, drug_term_mean = NA_real_,
                      null_sd = NA_real_, z = NA_real_, p = NA_real_,
                      drug_term = drug_term,
                      stringsAsFactors = FALSE))
  }

  s <- sig$sign[match(genes, sig$gene)]
  w <- sig$weight[match(genes, sig$gene)]
  e <- drug_effect[genes]

  # One scale for both the score and the null, so adding the null cannot move
  # the score.
  sc <- if (drug_term == "scaled") .robust_scale(e) else NA_real_
  d  <- if (drug_term == "sign") sign(e) else tanh(e / sc)

  # Global shift of the drug term over the whole signature, which is what
  # E[d_g] = 0 assumes away. Measured on every gene the drug reports, not just
  # the module overlap, because a shift is a property of the signature.
  all_e <- drug_effect[is.finite(drug_effect) & drug_effect != 0]
  d_ref <- if (length(all_e)) {
    if (drug_term == "sign") sign(all_e) else tanh(all_e / sc)
  } else d
  drug_term_mean <- mean(d_ref)
  if (center_drug_term) d <- d - drug_term_mean

  product   <- s * d                       # -1 = reversal, +1 = reinforcement
  score     <- -sum(w * product) / sum(w)  # negate so that + means reversal
  reversal  <- sum(sign(product) < 0)
  reinforce <- sum(sign(product) > 0)

  # --- analytic null -------------------------------------------------------
  # Var(d_g) under E[d_g] = 0 is E[d_g^2]. For the sign term that is exactly 1;
  # for the scaled term it is estimated from every gene the drug measured, using
  # the scale already fixed above.
  # Var(d_g) = E[d_g^2] under E[d_g] = 0. Exactly 1 for the sign term; for the
  # scaled term estimated PER SIGNATURE, never globally -- signatures differ in
  # overall response magnitude, and a pooled variance would flatter a weak
  # responder. Centering, when on, propagates into the variance too.
  var_d <- if (drug_term == "sign" && !center_drug_term) 1 else {
    ref <- if (length(all_e) >= n_overlap) d_ref else d
    if (center_drug_term) mean((ref - drug_term_mean)^2) else mean(ref^2)
  }
  weight_sum <- sum(w)
  n_eff      <- weight_sum^2 / sum(w^2)
  max_weight_share <- max(w) / weight_sum
  null_sd    <- sqrt(var_d / n_eff)
  z          <- if (is.finite(null_sd) && null_sd > 0) score / null_sd else NA_real_
  p          <- if (is.finite(z)) 2 * stats::pnorm(-abs(z)) else NA_real_

  data.frame(score = score, n_overlap = n_overlap,
             reversal_n = reversal, reinforce_n = reinforce,
             weight_sum = weight_sum, n_eff = n_eff,
             max_weight_share = max_weight_share, drug_term_mean = drug_term_mean,
             null_sd = null_sd, z = z, p = p,
             drug_term = drug_term,
             stringsAsFactors = FALSE)
}


# Batch form: score many drugs against many modules in one matrix multiply.
# Far faster than a loop for whole-library screens.
#
# IT AGREES WITH direction_score() ONLY WHEN THE DRUG MEASURES EVERY MODULE
# GENE, and that condition does not hold here. The two differ in what goes into
# the denominator when a drug measures only part of a module:
#
#   direction_score()        divides by the weight of the genes it actually
#                            scored, so the score is a weighted mean over the
#                            overlap. This is the pAGE / RGES convention.
#   direction_score_matrix() divides by the weight of every module gene, because
#                            an unmeasured gene enters the numerator as d_g = 0.
#                            An unmeasured gene is then treated as a measured
#                            null effect, which pulls the score toward zero in
#                            proportion to how much of the module the drug
#                            misses.
#
# With CMap landmark coverage the overlap runs from 5 to 18 genes per module, so
# the dilution is not a corner case. `denominator` selects between them:
#
#   "module"   (default) the behaviour every result to date was produced with,
#              kept as the default so nothing changes silently
#   "measured" the direction_score() convention; removes the dilution and makes
#              the two functions agree exactly
#
# Switching this changes every score, so it is a deliberate analysis decision,
# not a tuning knob. It was verified that "module" reproduces the previous
# implementation bit for bit.
#
#   sign_matrix   : genes x modules matrix of disease signs (-1/+1, 0 if the gene
#                   is not in that module's signature).
#   weight_matrix : genes x modules matrix of gene weights (0 outside the module).
#                   Pass a single genes-length vector to use the same weights for
#                   every module, or a matrix of ones for the pAGE baseline.
#   drug_matrix   : genes x drugs matrix of drug effects (e.g. CMap z-scores),
#                   dense over the measured genes.
#   drug_term     : "sign" or "scaled" (as above).
#   min_overlap   : per (module, drug) minimum overlap; scores below it become NA.
#   denominator   : "module" (default) or "measured"; see above.
#
# Returns a modules x drugs matrix of scores, carrying "n_overlap" and the
# analytic null as attributes "n_eff", "null_sd", "z" and "p" (each a
# modules x drugs matrix). They are attributes rather than a list so existing
# callers keep working.
direction_score_matrix <- function(sign_matrix, weight_matrix, drug_matrix,
                                   drug_term = c("sign", "scaled"),
                                   min_overlap = 3,
                                   denominator = c("module", "measured")) {
  drug_term   <- match.arg(drug_term)
  denominator <- match.arg(denominator)

  if (is.null(dim(weight_matrix))) {
    weight_matrix <- matrix(weight_matrix, nrow = length(weight_matrix),
                            ncol = ncol(sign_matrix),
                            dimnames = list(names(weight_matrix), colnames(sign_matrix)))
  }

  # align all three on the genes they share
  genes <- Reduce(intersect, list(rownames(sign_matrix),
                                  rownames(weight_matrix),
                                  rownames(drug_matrix)))
  S <- sign_matrix[genes, , drop = FALSE]
  W <- weight_matrix[genes, , drop = FALSE]
  D <- drug_matrix[genes, , drop = FALSE]

  # The robust scale for the "scaled" term is taken over the genes the drug
  # actually measured. Everything else in this function reads a zero as "not
  # measured" (see `measured` below), so letting those zeros into the MAD would
  # shrink the scale, saturate the tanh, and make the batch form disagree with
  # direction_score() on exactly the columns that carry unmeasured genes. Columns
  # that encode unmeasured genes as NA rather than 0 are unaffected either way.
  d <- if (drug_term == "sign") sign(D) else apply(D, 2, function(col) {
    ok <- is.finite(col) & col != 0
    if (!any(ok)) return(rep(0, length(col)))
    tanh(col / .robust_scale(col[ok]))
  })
  d[!is.finite(d)] <- 0

  present   <- (S != 0)                    # genes x modules
  measured  <- (d != 0)                    # genes x drugs
  n_overlap <- t(present) %*% measured     # modules x drugs

  WS <- W * S                              # genes x modules
  numerator <- t(WS) %*% d                 # modules x drugs
  # weight sum: over the module's genes, or over the genes the drug measured
  den <- if (denominator == "module") {
    matrix(colSums(abs(W) * present), nrow = ncol(S), ncol = ncol(d),
           dimnames = list(colnames(S), colnames(d)))
  } else {
    t(abs(W) * present) %*% measured
  }
  score <- -numerator / den
  score[n_overlap < min_overlap] <- NA_real_

  # --- analytic null, per (module, drug) -----------------------------------
  # A gene the drug did not measure contributes d_g = 0, so it adds nothing to
  # the numerator's variance while its weight still sits in the denominator.
  # The variance sum therefore runs over the MEASURED genes and the weight sum
  # over the module's genes:
  #     sigma = sqrt( var_d * sum_measured w^2 ) / sum_module w
  # which collapses to sqrt(var_d / N_eff) with N_eff = (sum w)^2 / sum w^2 when
  # the drug measures the whole module -- the scalar direction_score() case.
  w2_measured <- t((W * present)^2) %*% measured        # modules x drugs
  n_eff       <- den^2 / w2_measured
  n_eff[!is.finite(n_eff)] <- NA_real_

  n_measured <- colSums(measured)
  # Per-drug global shift of the drug term: what E[d_g] = 0 assumes is zero.
  drug_term_mean <- colSums(d) / pmax(n_measured, 1)
  # Largest single gene's share of the contributing weight, per (module, drug).
  # A share near 1 is a score resting on one gene, which is exactly where the
  # normal approximation at N_eff of 5 to 18 is least trustworthy.
  aw <- abs(W) * present
  max_share <- matrix(NA_real_, ncol(S), ncol(d),
                      dimnames = list(colnames(S), colnames(d)))
  for (mi in seq_len(ncol(S))) {
    wm <- aw[, mi]
    mx <- apply(measured, 2, function(msk) if (any(msk)) max(wm[msk]) else NA_real_)
    max_share[mi, ] <- mx / den[mi, ]
  }
  var_d <- if (drug_term == "sign") rep(1, ncol(d)) else {
    v <- colSums(d^2) / pmax(n_measured, 1)
    v[!is.finite(v) | v <= 0] <- NA_real_
    v
  }
  null_sd <- sqrt(sweep(1 / n_eff, 2, var_d, "*"))
  z <- score / null_sd
  z[!is.finite(z)] <- NA_real_
  p <- 2 * stats::pnorm(-abs(z))

  n_eff[is.na(score)] <- NA_real_
  null_sd[is.na(score)] <- NA_real_
  z[is.na(score)] <- NA_real_
  p[is.na(score)] <- NA_real_

  attr(score, "n_eff")   <- n_eff
  attr(score, "null_sd") <- null_sd
  attr(score, "z")       <- z
  attr(score, "p")       <- p
  attr(score, "max_weight_share") <- max_share
  attr(score, "drug_term_mean")   <- drug_term_mean
  attr(score, "n_overlap")   <- n_overlap
  attr(score, "denominator") <- denominator
  score
}


# Build a disease signature from a per-cell-type differential-expression table.
#
# This is the disease side of the cell-type-resolved score: s_g comes from the
# per-cell-type single-cell DE, d_g still comes from a perturbation reference.
# A per-cell-type DE is composition-free by construction, so it does not carry
# the cell-loss component that makes a bulk direction partly unreachable by any
# compound.
#
# The score stays only HALF cell-type-specific and that must be stated wherever
# it is reported: the disease direction is resolved by cell type, the drug
# direction is a cell line that is not a neuron. There is no dopaminergic neuron
# in CMap, and the cell-type biology is on the disease side where it can be
# resolved.
#
#   de_table   : data.frame with gene, cell_type, logfc and (optionally) pvalue,
#                fdr -- the `de_table_symbol` emitted by process_singlecell()
#   cell_type  : which cell type to build the signature for
#   genes      : optional restriction to the module's genes
#   weight     : "none"          -> w_g = 1 (the pAGE-equivalent weighting)
#                "one_minus_fdr" -> w_g = 1 - fdr, confidence without exclusion
#                "neglog10p"     -> w_g = -log10(p), rescaled to a max of 1
#
# NO FDR THRESHOLD IS APPLIED, deliberately. Per-cell-type pseudobulk DE with
# few donors has low power, and the module-by-landmark overlap is already the
# binding constraint at 5 to 18 genes. Every gene with a direction contributes
# its sign; confidence enters through w_g, and n_eff then reports how much
# evidence the result actually rests on.
disease_signature_from_de <- function(de_table, cell_type, genes = NULL,
                                      weight = c("none", "one_minus_fdr", "neglog10p")) {
  weight <- match.arg(weight)
  de <- as.data.frame(de_table, stringsAsFactors = FALSE)
  stopifnot("de_table needs gene, cell_type and logfc columns" =
              all(c("gene", "cell_type", "logfc") %in% names(de)))

  de <- de[de$cell_type == cell_type, , drop = FALSE]
  if (!is.null(genes)) de <- de[de$gene %in% unique(as.character(genes)), , drop = FALSE]
  de <- de[is.finite(de$logfc) & de$logfc != 0, , drop = FALSE]
  if (!nrow(de))
    return(data.frame(gene = character(), sign = numeric(), weight = numeric(),
                      stringsAsFactors = FALSE))

  w <- switch(weight,
    none = rep(1, nrow(de)),
    one_minus_fdr = {
      f <- if (is.null(de$fdr)) rep(0, nrow(de)) else de$fdr
      f[!is.finite(f)] <- 1
      pmax(1 - f, 0)
    },
    neglog10p = {
      pv <- if (is.null(de$pvalue)) rep(NA_real_, nrow(de)) else de$pvalue
      pv[!is.finite(pv)] <- 1
      v <- -log10(pmax(pv, 1e-300))
      if (max(v) > 0) v / max(v) else rep(1, length(v))
    })

  out <- data.frame(gene = as.character(de$gene), sign = sign(de$logfc),
                    weight = as.numeric(w), stringsAsFactors = FALSE)
  # a zero weight is a silently dropped gene; keep it visible as a small floor
  out <- out[out$weight > 0, , drop = FALSE]
  out[!duplicated(out$gene), , drop = FALSE]
}
