# =============================================================================
# NetworkProximity : load_reference.R
#
# Reads the reference data from plain, swappable files (no bundled .rda). Every
# input is a gene-SYMBOL table, so a user can drop in their own PPI or DTI by
# matching these simple formats.
#
#   PPI edge file  : two columns of gene symbols (an undirected edge per row).
#   DTI file       : the drug-target table. Column names are given as arguments
#                    so any DTI layout works; the defaults match the project's
#                    master table.
#   blacklist file : two columns of drug ids (an adverse pair per row).
#
# The last section holds the adapters between the symbol-native steps (00, 02,
# 04) and the Entrez-era steps (01, 03, 05), which were migrated at different
# times and therefore disagree on column names.
# =============================================================================

suppressPackageStartupMessages({ library(data.table); library(igraph) })


# Build an undirected igraph from a two-column edge file of gene symbols.
load_ppi <- function(ppi_path, col_a = 1, col_b = 2, header = TRUE) {
  e <- fread(ppi_path, header = header)
  e <- e[, c(col_a, col_b), with = FALSE]
  setnames(e, c("from", "to"))
  e[, from := trimws(as.character(from))][, to := trimws(as.character(to))]
  e <- unique(e[from != "" & to != "" & from != to])
  g <- graph_from_data_frame(e, directed = FALSE)
  simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
}


# Read the drug-target table into the symbol-native columns the model expects:
#   dti             : drug, drug_name, drug_type, target_gene
#   drug_structures : drug, drug_name, drug_type, inchikey, smiles, has_structure
# Column names in the source file are arguments (defaults match the master table).
load_dti <- function(dti_path,
                     drug_col = "DrugBank_ID", name_col = "Drug_Name_links",
                     type_col = "Drug_Type_links", gene_col = "Gene_Symbol",
                     inchikey_col = "InChIKey", smiles_col = "SMILES") {
  x <- fread(dti_path)
  dti <- unique(data.table(
    drug        = trimws(as.character(x[[drug_col]])),
    drug_name   = as.character(x[[name_col]]),
    drug_type   = as.character(x[[type_col]]),
    target_gene = trimws(as.character(x[[gene_col]]))))
  dti <- dti[drug != "" & target_gene != ""]

  drug_structures <- unique(data.table(
    drug      = trimws(as.character(x[[drug_col]])),
    drug_name = as.character(x[[name_col]]),
    drug_type = as.character(x[[type_col]]),
    inchikey  = if (inchikey_col %in% names(x)) as.character(x[[inchikey_col]]) else NA_character_,
    smiles    = if (smiles_col   %in% names(x)) as.character(x[[smiles_col]])   else NA_character_))
  drug_structures <- drug_structures[!duplicated(drug)]
  drug_structures[, has_structure := !is.na(smiles) & smiles != ""]
  # build_drug_contract() needs is_small_molecule to tell a genuine
  # virtual-screening miss from a drug no structure-based model can score at all.
  # It is derivable from drug_type, so derive it here rather than making every
  # caller remember to add the column.
  drug_structures[, is_small_molecule := .is_small_molecule(drug_type)]

  list(dti = dti, drug_structures = drug_structures)
}


# DrugBank's drug_type is "small molecule" or "biotech"; anything that is not
# recognisably a small molecule (including a missing type) is treated as not one,
# because the cost of a wrong TRUE is counting an antibody as an AI-component miss.
.is_small_molecule <- function(drug_type) {
  ty <- tolower(trimws(as.character(drug_type)))
  !is.na(ty) & grepl("small[ _-]?molecule", ty)
}


# Read the adverse drug-pair blacklist into columns drug_a, drug_b.
load_blacklist <- function(blacklist_path, col_a = 1, col_b = 2, header = TRUE) {
  b <- fread(blacklist_path, header = header)
  b <- b[, c(col_a, col_b), with = FALSE]
  setnames(b, c("drug_a", "drug_b"))
  b[, drug_a := trimws(as.character(drug_a))][, drug_b := trimws(as.character(drug_b))]
  unique(b[drug_a != "" & drug_b != ""])
}


# =============================================================================
# Adapters: symbol-native steps (00, 02, 04) -> Entrez-era steps (01, 03, 05)
#
# Steps 00, 02 and 04 are symbol-native: graph nodes, module genes and drug
# targets are all gene symbols, and the drug id column is `drug`. Steps 01, 03
# and 05 are the retained context (dynamic) extension and are still Entrez-keyed
# internally, expecting `drugbank_id`, `target_entrez_id`, `gene_symbol` and an
# `entrez_id` node key.
#
# Rather than half-migrate the dynamic extension -- which would invalidate the
# runs it has already produced -- the two conventions are bridged here. The
# functions below are pure renames plus one identity map; none of them changes a
# statistic. output.R additionally accepts either drug id column directly.
# =============================================================================


# Run the Entrez-era steps symbol-natively.
#
# score_genes() maps symbols through `symbol_to_entrez` and keys everything
# downstream on the result. Passing an identity map makes that key the symbol
# itself, so gene_scores$entrez_id, the PPI node names and the DTI target column
# all live in one namespace and the dynamic steps run on a symbol-keyed PPI with
# no Entrez lookup anywhere. Use this when the PPI came from load_ppi().
identity_symbol_map <- function(symbols) {
  s <- unique(trimws(as.character(symbols)))
  s <- s[!is.na(s) & s != ""]
  data.table(symbol = s, entrez_id = s)
}


# load_dti()$dti -> the column names build_node_weights() expects.
# `target_entrez_id` carries symbols when the component is run symbol-natively (see
# identity_symbol_map); the column name is historical, not a claim about the
# id namespace.
dti_for_dynamic <- function(dti) {
  d <- as.data.table(dti)
  stopifnot("dti needs `drug` and `target_gene` columns" =
              all(c("drug", "target_gene") %in% names(d)))
  unique(data.table(
    drugbank_id      = as.character(d$drug),
    drug_name        = as.character(d$drug_name),
    target_entrez_id = as.character(d$target_gene),
    gene_symbol      = as.character(d$target_gene)))
}


# load_ppi() graph -> the two-column edge table build_dynamic_graphs() expects.
ppi_edges_from_graph <- function(graph) {
  stopifnot("pass an igraph" = inherits(graph, "igraph"))
  e <- as.data.table(igraph::as_data_frame(graph, what = "edges"))
  setnames(e, names(e)[1:2], c("from", "to"))
  unique(e[, .(from = as.character(from), to = as.character(to))])
}
