# =============================================================================
# FunctionalDirection : perturbation_reference.R
#
# Loads the drug-perturbation reference (CMap / LINCS Level-5) into a compact,
# gene-symbol-keyed object that the direction scoring reads from. The big .gctx
# file is read once here for the genes and instances we care about; everything
# downstream works on the small in-memory matrix.
#
# Reading logic follows the LINCS Level-5 .gctx layout and is unchanged from the
# verified original.
# =============================================================================


# Read the row (gene) and column (signature) identifiers from a .gctx file.
read_gctx_ids <- function(gctx_path) {
  list(row_ids = as.character(rhdf5::h5read(gctx_path, "/0/META/ROW/id")),
       col_ids = as.character(rhdf5::h5read(gctx_path, "/0/META/COL/id")))
}


# Read a sub-matrix of the .gctx: the requested gene ids (rows) x signature ids
# (columns). Returns a matrix with gene-id rownames and sig-id colnames.
read_gctx_matrix <- function(gctx_path, gene_ids, sig_ids) {
  ids <- read_gctx_ids(gctx_path)
  row_idx <- match(as.character(gene_ids), ids$row_ids)
  col_idx <- match(as.character(sig_ids),  ids$col_ids)
  keep_r <- !is.na(row_idx)
  keep_c <- !is.na(col_idx)
  if (!any(keep_r) || !any(keep_c)) return(matrix(numeric(), 0, 0))

  row_order <- order(row_idx[keep_r])
  col_order <- order(col_idx[keep_c])
  mat <- rhdf5::h5read(gctx_path, "/0/DATA/0/matrix",
                       index = list(row_idx[keep_r][row_order],
                                    col_idx[keep_c][col_order]))
  mat <- as.matrix(mat)
  rownames(mat) <- as.character(gene_ids[keep_r][row_order])
  colnames(mat) <- as.character(sig_ids[keep_c][col_order])
  # restore the caller's requested order
  mat[as.character(gene_ids[keep_r]), as.character(sig_ids[keep_c]), drop = FALSE]
}


# Pick one signature instance per drug in a given cellular context.
# General: the context (cell line), dose unit and quality column are all
# arguments, so the same function serves any cell line.
#   siginfo      : the LINCS signature metadata table
#   drug_map     : data.frame(drug, pert_id) linking your drug ids to CMap pert ids
#   cell_line    : e.g. "MCF7", "ASC", "NPC"
#   dose_unit    : e.g. "uM"
#   quality_col  : signature-quality column; the highest value is chosen
# Returns one row per drug: the chosen sig_id and its metadata.
select_perturbation_instances <- function(siginfo, drug_map, cell_line,
                                          dose_unit = "uM", quality_col = "cc_q75") {
  si <- as.data.frame(siginfo)
  if (!quality_col %in% names(si)) quality_col <- "distil_cc_q75"
  si$dose_num <- suppressWarnings(as.numeric(si$pert_dose))
  si$quality  <- suppressWarnings(as.numeric(si[[quality_col]]))
  si$quality[is.na(si$quality)] <- -Inf

  keep <- si$pert_type == "trt_cp" & si$pert_dose_unit == dose_unit &
          si$cell_mfc_name == cell_line & is.finite(si$dose_num)
  if ("qc_pass" %in% names(si)) keep <- keep & si$qc_pass == 1
  si <- si[keep, , drop = FALSE]

  pert_ids <- unique(drug_map$pert_id)
  si$mapped_pert <- ifelse(si$pert_mfc_id %in% pert_ids, si$pert_mfc_id, si$pert_id)
  si <- si[si$mapped_pert %in% pert_ids, , drop = FALSE]
  if (!nrow(si)) return(si[0, , drop = FALSE])

  si <- merge(drug_map, si, by.x = "pert_id", by.y = "mapped_pert")
  # highest dose, then highest quality, per drug
  si <- si[order(si$drug, -si$dose_num, -si$quality), , drop = FALSE]
  chosen <- si[!duplicated(si$drug), , drop = FALSE]
  chosen$cell_line <- cell_line
  chosen
}


# Build the perturbation reference object for one or more cellular contexts.
#   gctx_path   : path to the Level-5 .gctx file
#   siginfo     : signature metadata table
#   geneinfo    : gene metadata (must have gene_id and gene_symbol)
#   drug_map    : data.frame(drug, pert_id)
#   genes       : gene symbols to keep (e.g. the union of all module genes)
#   contexts    : character vector of cell lines to load
#   dose_unit, quality_col : passed to select_perturbation_instances()
#
# Returns a list with, per context: the chosen instances and a gene-symbol x
# instance expression matrix. Read the big file once; use the object many times.
prepare_perturbation_reference <- function(gctx_path, siginfo, geneinfo, drug_map,
                                          genes, contexts,
                                          dose_unit = "uM", quality_col = "cc_q75") {
  gi <- as.data.frame(geneinfo)
  gi$gene_id <- as.character(gi$gene_id)
  want <- gi[gi$gene_symbol %in% genes, c("gene_id", "gene_symbol")]
  want <- want[!duplicated(want$gene_id), ]

  per_context <- list()
  for (ctx in contexts) {
    inst <- select_perturbation_instances(siginfo, drug_map, ctx, dose_unit, quality_col)
    if (!nrow(inst)) { per_context[[ctx]] <- list(instances = inst, matrix = NULL); next }
    mat <- read_gctx_matrix(gctx_path, want$gene_id, inst$sig_id)
    if (nrow(mat)) rownames(mat) <- want$gene_symbol[match(rownames(mat), want$gene_id)]
    per_context[[ctx]] <- list(instances = inst, matrix = mat)
  }
  list(contexts = contexts, genes = genes, per_context = per_context)
}
