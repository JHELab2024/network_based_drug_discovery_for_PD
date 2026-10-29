# =============================================================================
# FunctionalDirection : drug_registry.R
#
# One drug lives under three different identifiers: a DrugBank id (used by the
# network component's drug-target table), a CMap pert id (used by the perturbation
# matrix), and a chemical structure (used for similarity search). The only thing
# they share is the structure, so the registry joins them through the InChIKey.
#
# The InChIKey has three blocks:  AAAAAAAAAAAAAA - BBBBBBBB - C
#   block 1 (14 chars) = molecular connectivity (the "skeleton")
#   block 2            = stereochemistry
#   block 3            = protonation / charge
# Matching on the full key is strict; matching on the skeleton (block 1) lets a
# salt (e.g. a hydrochloride) match its free base, which recovers many more
# drug <-> CMap links. We keep both and record which was used.
#
# This builder is pure R and uses the InChIKey columns already present in the
# inputs. (Optional structure standardisation with RDKit can refine the SMILES
# before fingerprinting, but is not required to build the registry.)
# =============================================================================


normalize_name <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- gsub("[^a-z0-9]+", " ", x)
  trimws(gsub("\\s+", " ", x))
}

inchikey_skeleton <- function(inchikey) {
  k <- toupper(trimws(as.character(inchikey)))
  ifelse(nchar(k) >= 14, substr(k, 1, 14), NA_character_)
}


# Build the drug registry linking DrugBank ids to CMap pert ids through structure.
#   drugs        : data.frame with columns drug (DrugBank id), name, inchikey, smiles
#   compoundinfo : CMap compound metadata with pert_id, inchi_key, cmap_name,
#                  and (optionally) compound_aliases
# Returns one row per DrugBank drug with its CMap pert id (NA if none), the match
# method, and coverage flags.
build_drug_registry <- function(drugs, compoundinfo) {
  d <- as.data.frame(drugs, stringsAsFactors = FALSE)
  d <- d[!duplicated(d$drug), , drop = FALSE]
  d$key_full <- toupper(trimws(d$inchikey))
  d$key_skel <- inchikey_skeleton(d$inchikey)
  d$name_norm <- normalize_name(d$name)

  ci <- as.data.frame(compoundinfo, stringsAsFactors = FALSE)
  ci$key_full <- toupper(trimws(ci$inchi_key))
  ci$key_skel <- inchikey_skeleton(ci$inchi_key)
  ci$name_norm <- normalize_name(ci$cmap_name)

  # helper: for the drugs still unmatched, join CMap on `by`, keep one-to-one hits
  first_match <- function(unmatched, drug_key, cmap_key) {
    du <- d[d$drug %in% unmatched & !is.na(d[[drug_key]]) & d[[drug_key]] != "", ]
    cu <- ci[!is.na(ci[[cmap_key]]) & ci[[cmap_key]] != "", c("pert_id", cmap_key)]
    cu <- cu[!duplicated(cu[[cmap_key]]), ]   # keep first CMap compound per key
    m <- merge(du[, c("drug", drug_key)], cu, by.x = drug_key, by.y = cmap_key)
    m <- m[!duplicated(m$drug), c("drug", "pert_id")]
    m
  }

  registry <- data.frame(drug = d$drug, pert_id = NA_character_,
                         match_method = NA_character_, stringsAsFactors = FALSE)
  steps <- list(c(key = "key_full",  method = "inchikey_full"),
                c(key = "key_skel",  method = "inchikey_skeleton"),
                c(key = "name_norm", method = "name"))
  for (st in steps) {
    todo <- registry$drug[is.na(registry$pert_id)]
    if (!length(todo)) break
    hit <- first_match(todo, st["key"], st["key"])
    if (nrow(hit)) {
      idx <- match(hit$drug, registry$drug)
      registry$pert_id[idx]      <- hit$pert_id
      registry$match_method[idx] <- st["method"]
    }
  }

  registry <- merge(registry,
                    d[, c("drug", "name", "inchikey", "smiles", "key_skel")],
                    by = "drug", all.x = TRUE)
  names(registry)[names(registry) == "key_skel"] <- "inchikey_skeleton"
  registry$has_structure <- !is.na(registry$smiles) & registry$smiles != ""
  registry$has_cmap      <- !is.na(registry$pert_id)
  registry[order(registry$drug), ]
}
