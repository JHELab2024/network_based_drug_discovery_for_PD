# =============================================================================
# AI (virtual screening) component — R binding over the `vecnet` Python package.
#
# This is the surface the combined R package will expose. The user gives four
# things and nothing else:
#
#   run_virtual_screening(targets, library, device = "cuda", finetune = FALSE)
#
#   targets  : character vector of protein sequences (named = target labels),
#              one or many
#   library  : character vector of SMILES (named = compound ids)
#   device   : "cuda" | "cpu" | NULL (auto-detect, falls back to CPU)
#   finetune : FALSE = pretrained baseline; TRUE = adapt to `library` first
#
# Tiers name the same choice more precisely: "0a" is the pretrained port, "0b" the
# library-weighted adaptation, "1" adds measured ChEMBL actives and is not built
# here -- asking for it records the request, runs 0b and says so in the report.
#
# Two Python-side defaults changed and are visible here. The fine-tune no longer
# runs a fixed five epochs (early stopping on a held-out weighted validation loss
# picks the epoch, and the pretrained model is inside that search space), and the
# library now *weights* the training pairs instead of filtering them through a
# min/max box.
#
# Asset paths are resolved from a single root, so they never reach the user.
# Lifting this into the package later means moving these functions into R/ and
# turning `vecnet_assets_root()` into a package option — no signature change.
# =============================================================================

.vecnet = new.env(parent = emptyenv())

# Locate the component beside this file; options can override any path.
.vecnet$root = local({
  for (frame in rev(sys.frames())) {
    if (!is.null(frame$ofile))
      return(dirname(dirname(normalizePath(frame$ofile, mustWork = TRUE))))
  }
  stop("Load this component with source('/path/to/core/VirtualScreening/R/vecnet_component.R').")
})

vecnet_assets_root = function() {
  getOption("vecnet.assets_root", .vecnet$root)
}

vecnet_module_path = function() {
  getOption("vecnet.module_path", .vecnet$root)
}

vecnet_python = function() {
  getOption("vecnet.python", file.path(dirname(dirname(.vecnet$root)), ".venv",
            if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python"))
}

# Preload torch's *bundled* CUDA libraries with RTLD_GLOBAL before torch is imported.
#
# Why: pip's torch ships its own nvidia libs, but if the machine also has system
# copies in the ldconfig cache (/lib/x86_64-linux-gnu/libnvJitLink.so.12 here), the
# system copy resolves first and torch's bundled libcusparse fails with
#   undefined symbol: __nvJitLinkAddData_12_1
# Plain `python` escapes this because torch preloads these itself; under R the import
# dies. LD_LIBRARY_PATH cannot fix it from inside R -- glibc reads it once at process
# start -- so we dlopen the correct library first and let its symbols win.
#
# Derived from the interpreter path, never hardcoded; a no-op on CPU-only installs.
.vecnet_preload_cuda = function(python) {
  if (.Platform$OS.type != "unix") return(invisible(character()))
  nv = Sys.glob(file.path(dirname(dirname(python)), "lib", "python*",
                          "site-packages", "nvidia", "nvjitlink", "lib",
                          "libnvJitLink.so.*"))
  if (!length(nv)) return(invisible(character()))
  ok = tryCatch({ dyn.load(nv[1], local = FALSE, now = FALSE); TRUE },
                error = function(e) FALSE)
  invisible(if (ok) nv[1] else character())
}

#' Bind the Python side. Called lazily; safe to call repeatedly.
#'
#' reticulate binds ONE interpreter per session, so this must run before anything
#' else touches Python. The interpreter needs torch + rdkit + gensim.
#'
#' Why this is more than a use_python() call: `RETICULATE_PYTHON` (commonly set in
#' ~/.Renviron) **silently overrides** use_python(), so an interpreter without torch
#' wins and the component dies with a bare "No module named 'torch'". This machine is in
#' exactly that state. We therefore set RETICULATE_PYTHON ourselves before Python is
#' initialised, and if Python is already bound to an unusable interpreter we say so
#' in terms the caller can act on rather than failing at import.
vecnet_init = function(python = vecnet_python(), module_path = vecnet_module_path()) {
  if (!is.null(.vecnet$module)) return(invisible(.vecnet$module))
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("the AI component needs the 'reticulate' package")

  # KMP_DUPLICATE_LIB_OK: torch and rdkit each ship an OpenMP runtime; without
  # this the second import aborts the R session rather than raising.
  if (Sys.getenv("KMP_DUPLICATE_LIB_OK") == "") Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

  already = reticulate::py_available(initialize = FALSE)
  if (!already) {
    if (!file.exists(python))
      stop("python interpreter not found: ", python,
           "\nRun source('install.R') from the release directory, or set options(vecnet.python = \"/path/to/python\").")
    # Must beat RETICULATE_PYTHON, which use_python() cannot override.
    Sys.setenv(RETICULATE_PYTHON = python)
    reticulate::use_python(python, required = TRUE)
  }
  .vecnet_preload_cuda(python)   # before the first `import torch`

  if (!reticulate::py_module_available("torch")) {
    cfg = reticulate::py_config()
    stop("the bound Python has no 'torch': ", cfg$python,
         if (already) "\nPython was already initialised in this session by another package; restart R and call vecnet_init() first."
         else paste0("\nRETICULATE_PYTHON=", Sys.getenv("RETICULATE_PYTHON")),
         "\nthe AI component needs an env with torch + rdkit + gensim (e.g. ", python, ")")
  }

  .vecnet$module = reticulate::import_from_path("vecnet", path = module_path)
  invisible(.vecnet$module)
}

#' Which interpreter is the AI component using, and does it have what it needs?
vecnet_diagnose = function() {
  vecnet_init()
  cfg = reticulate::py_config()
  mods = c("torch", "rdkit", "gensim", "numpy", "pandas", "sklearn")
  data.frame(component = c("python", "module_path", mods),
             value = c(cfg$python, vecnet_module_path(),
                       vapply(mods, function(m)
                         as.character(reticulate::py_module_available(m)), character(1))),
             row.names = NULL, stringsAsFactors = FALSE)
}

#' Is a CUDA device actually usable from this session?
vecnet_cuda_available = function() {
  vecnet_init()
  isTRUE(.vecnet$module$cuda_available())
}

#' Read protein sequences from a FASTA file -> named character vector.
vecnet_read_fasta = function(path) {
  vecnet_init()
  unlist(.vecnet$module$read_fasta(normalizePath(path, mustWork = TRUE)))
}

# Values and labels cross to Python as two parallel lists, never as one dict.
#
# A dict looks tidier but is wrong here: duplicate keys collapse, so two compounds
# sharing an id would silently become one and the library would come back shorter
# than it went in -- with no error. Parallel lists preserve order, length and
# duplicates. (Note `as.character()` drops names, so labels are taken first.)
.as_py_values = function(x) as.list(unname(as.character(x)))

# reticulate maps R NULL to Python None, and every Python argument that takes NULL
# reads it as "use the default" -- so an unset option never becomes an as.integer()
# of nothing.
.int_or_null = function(x) if (is.null(x)) NULL else as.integer(x)

.as_py_labels = function(x) {
  nm = names(x)
  if (is.null(nm) || !all(nzchar(nm))) return(NULL)   # unnamed -> Python labels them
  as.list(as.character(nm))
}

#' Screen a compound library against one or many protein targets.
#'
#' @param targets  character vector of AA sequences; names become target labels
#' @param library  character vector of SMILES; names become compound ids
#' @param device   "cuda", "cpu", or NULL to auto-detect (CPU fallback)
#' @param finetune FALSE -> pretrained cold-start model (baseline)
#'                 TRUE  -> fine-tune drug_proj+head to `library` first, once,
#'                          then reuse that model for every target
#' @param tuned_dir optional dir to persist the tuned bundle (finetune = TRUE)
#' @param tier     NULL (derive from `finetune`), "0a", "0b" or "1"; an explicit
#'                 tier overrides `finetune`, and "1" falls back to "0b" with the
#'                 reason recorded in the report
#' @param finetune_epochs NULL = let early stopping choose (the default since the
#'                 fixed five epochs were dropped); an integer pins a maximum
#' @param finetune_max_epochs ceiling on that search, not a target
#' @param subspace_mode NULL/"density_ratio" = weight the training pairs by how much
#'                 they resemble `library`; "legacy_box" = the pre-A1 min/max filter
#' @param calibrate TRUE -> also return the library percentile and the heavy-atom
#'                 residual, which are what make scores comparable across targets
#' @param reference_ligand a known binder -- SMILES, or a mol2/sdf path -- turning on
#'                 the ligand-similarity channel; a named list gives one per target
#' @param similarity_band resolution of that channel. NULL (the default) chooses it
#'                 per target from `known_binders`; a number fixes it, with 0 meaning
#'                 similarity alone and 1 the sequence ranking exactly
#' @param max_references cap on how many binders are used as similarity queries.
#'                 Cost is linear in this: 1.82M compounds against 1,000 references
#'                 takes about 4 minutes, against 11,000 about 45
#' @param known_binders compounds already known to bind, as a character vector or a
#'                 named list per target. Supplying them turns the similarity channel
#'                 on and lets the band be calibrated; they double as the references
#'                 when no `reference_ligand` is given, and are never ranked.
#' @param long     TRUE -> tidy one-row-per-(compound, target) data.frame;
#'                 FALSE -> compounds x targets score matrix
#' @return see `long`; attribute "report" carries the run summary
run_virtual_screening = function(targets, library, device = NULL, finetune = FALSE,
                                 tuned_dir = NULL, long = TRUE, verbose = FALSE,
                                 assets_root = vecnet_assets_root(),
                                 finetune_epochs = NULL, finetune_lr = 1e-4,
                                 finetune_max_epochs = NULL, batch_size = 4096L,
                                 tier = NULL, subspace_mode = NULL,
                                 calibrate = FALSE,
                                 reference_ligand = NULL, similarity_band = NULL,
                                 known_binders = NULL, calibration_fraction = 0.01,
                                 max_references = 1000L) {
  vecnet_init()
  stopifnot(length(targets) > 0, length(library) > 0,
            is.logical(finetune), length(finetune) == 1)

  out = .vecnet$module$screen(
    targets             = .as_py_values(targets),
    target_names        = .as_py_labels(targets),
    library             = .as_py_values(library),
    library_ids         = .as_py_labels(library),
    assets_root         = assets_root,
    device              = device,
    finetune            = finetune,
    tier                = tier,
    tuned_dir           = tuned_dir,
    verbose             = verbose,
    batch_size          = as.integer(batch_size),
    finetune_epochs     = .int_or_null(finetune_epochs),
    finetune_max_epochs = .int_or_null(finetune_max_epochs),
    finetune_lr         = finetune_lr,
    subspace_mode       = subspace_mode,
    calibrate           = calibrate,
    reference_ligand    = reference_ligand,
    similarity_band     = similarity_band,
    known_binders       = known_binders,
    calibration_fraction = calibration_fraction,
    max_references      = .int_or_null(max_references)
  )

  scores = out$scores                       # numpy (n_compounds x n_targets) -> matrix
  dimnames(scores) = list(unlist(out$ids), unlist(out$targets))
  report = out$report

  if (!long) {
    attr(scores, "report") = report
    return(scores)
  }

  res = data.frame(
    compound_id = rep(rownames(scores), times = ncol(scores)),
    smiles      = rep(unlist(out$smiles), times = ncol(scores)),
    target      = rep(colnames(scores), each = nrow(scores)),
    score       = as.vector(scores),
    stringsAsFactors = FALSE
  )
  # The calibrated columns travel beside the raw score rather than replacing it:
  # the raw score is what the model said, `percentile` is what it means relative to
  # this target's own library distribution, and `size_residual` is what is left once
  # molecular size is taken out. Rank on the last of the three.
  if (!is.null(out$percentile)) {
    res$percentile    = as.vector(out$percentile)
    res$size_residual = as.vector(out$size_residual)
  }
  # The ligand-similarity channel, when a reference ligand was supplied. `combined`
  # is what to rank on: similarity decides the ordering and the sequence model only
  # separates compounds that similarity cannot tell apart.
  if (!is.null(out$similarity)) {
    res$similarity = as.vector(out$similarity)
    res$combined   = as.vector(out$combined)
    # With similarity_band = "auto" the band is a per-target quantity, so it travels
    # with the rows it governs: 1 means the channel was switched off for that target
    # and `combined` is the sequence ranking.
    if (!is.null(out$similarity_band)) {
      bands = vapply(colnames(scores), function(tn) out$similarity_band[[tn]]$band,
                     numeric(1))
      res$similarity_band = rep(unname(bands), each = nrow(scores))
    }
  }
  if (!is.null(out$fold_spread)) res$fold_sd = as.vector(out$fold_spread)
  attr(res, "report") = report
  res
}


#' Bundled reference data for the Parkinson's disease targets used in the paper.
#'
#' The ligand-similarity channel needs a known binder to compare against and enough
#' further binders to calibrate the band. Both are public but assembling them is the
#' step most likely to stop someone trying the channel, so they ship with the package.
#'
#'   ref = vecnet_reference_binders("LRRK2")
#'   run_virtual_screening(targets = ref$targets, library = my_library,
#'                         reference_ligand = ref$reference_ligand,
#'                         known_binders    = ref$known_binders,
#'                         similarity_band  = "auto")
#'
#' The band itself is not bundled: it is a property of the library being screened,
#' not of the target, so it is chosen at run time against `my_library`.
#'
#' @param symbol one target, a character vector of them, or NULL for all
#' @param min_pchembl keep only binders at or above this potency (NULL keeps all)
#' @return list(targets, reference_ligand, known_binders), each named by target
vecnet_reference_binders = function(symbol = NULL, min_pchembl = NULL) {
  vecnet_init()
  .vecnet$module$reference_binders(symbol = symbol, min_pchembl = min_pchembl)
}

#' Which targets the bundled reference data covers.
vecnet_reference_targets = function() {
  vecnet_init()
  do.call(rbind, lapply(.vecnet$module$reference_targets(), function(r)
    data.frame(symbol = r$symbol, uniprot = r$uniprot, pdb = r$pdb,
               het_code = r$het_code, length_aa = r$length_aa,
               n_binders = r$n_binders, stringsAsFactors = FALSE)))
}


#' Top-n compounds for one target from a run_virtual_screening(long = TRUE) result.
vecnet_top_hits = function(result, target = NULL, n = 20L) {
  stopifnot(is.data.frame(result))
  if (is.null(target)) target = result$target[1]
  sub = result[result$target == target & !is.na(result$score), ]
  sub[order(-sub$score), ][seq_len(min(n, nrow(sub))), ]
}

# =============================================================================
# A5 -- the evaluation primitives, as R.
#
# `evaluate.py` is deliberately a set of primitives: it defines the split, the decoy
# rule and the metric once, and leaves "which actives, which library, which targets"
# to the analysis driver. Our driver is R, so these are the four calls it needs. They
# add no logic of their own -- each is an argument translation and a 0-based to
# 1-based index shift, so the metric an R script reports and the metric a Python
# script reports are the same function.
# =============================================================================

#' Property-matched decoys for one target's actives, drawn from a candidate library.
#'
#' Nearest neighbours in z-scored property space (size, logP, flexibility, H-bonding,
#' polarity), minus anything sharing a Murcko scaffold with an active and anything
#' within `max_tanimoto` of one. Enrichment against random library compounds measures
#' drug-likeness; enrichment against these measures recognition.
#'
#' @return list(index = 1-based positions into `candidate_smiles`, report = run summary)
vecnet_property_matched_decoys = function(active_smiles, candidate_smiles,
                                          n_per_active = 50L, exclude_scaffolds = TRUE,
                                          max_tanimoto = 0.35, max_candidates = 200000L,
                                          seed = 0L) {
  vecnet_init()
  stopifnot(length(active_smiles) > 0, length(candidate_smiles) > 0)
  out = .vecnet$module$property_matched_decoys(
    active_smiles     = .as_py_values(active_smiles),
    candidate_smiles  = .as_py_values(candidate_smiles),
    n_per_active      = as.integer(n_per_active),
    exclude_scaffolds = exclude_scaffolds,
    max_tanimoto      = max_tanimoto,
    max_candidates    = .int_or_null(max_candidates),
    seed              = as.integer(seed))
  list(index = as.integer(out[[1]]) + 1L, report = out[[2]])
}

#' Early-recognition metrics for one target's ranking.
#'
#' `size_corrected = TRUE` ranks on the heavy-atom residual instead of the raw score.
#' Both are meant to be reported: enrichment that survives the correction is about the
#' chemistry, and enrichment that does not is the size artifact.
vecnet_enrichment_report = function(scores, labels, smiles = NULL, heavy_atoms = NULL,
                                    fractions = c(0.001, 0.005, 0.01, 0.05), alpha = 20,
                                    tie_break = "random", seed = 0L,
                                    size_corrected = FALSE, residual_method = "linear") {
  vecnet_init()
  stopifnot(length(scores) == length(labels))
  .vecnet$module$enrichment_report(
    scores          = as.numeric(scores),
    labels          = as.numeric(labels),
    smiles          = if (is.null(smiles)) NULL else .as_py_values(smiles),
    heavy_atoms     = if (is.null(heavy_atoms)) NULL else as.numeric(heavy_atoms),
    fractions       = as.list(as.numeric(fractions)),
    alpha           = as.numeric(alpha),
    tie_break       = tie_break,
    seed            = as.integer(seed),
    size_corrected  = size_corrected,
    residual_method = residual_method)
}

#' Murcko scaffold groups -> named list of 1-based index vectors, one per scaffold.
#' Vectorised on the Python side; the per-compound form would pay reticulate's call
#' overhead once per molecule.
vecnet_scaffold_groups = function(smiles, include_chirality = FALSE, generic = FALSE,
                                  acyclic = "group") {
  vecnet_init()
  groups = .vecnet$module$evaluate$scaffold_groups(
    .as_py_values(smiles), include_chirality = include_chirality,
    generic = generic, acyclic = acyclic)
  lapply(groups, function(idx) as.integer(idx) + 1L)
}

#' Heavy-atom count per compound; nan where RDKit cannot parse the SMILES.
#' The input to A2's size correction, exposed so it can be computed once for a library
#' and reused across every target rather than re-parsed per target.
vecnet_heavy_atoms = function(smiles) {
  vecnet_init()
  as.numeric(.vecnet$module$heavy_atom_counts(.as_py_values(smiles)))
}
