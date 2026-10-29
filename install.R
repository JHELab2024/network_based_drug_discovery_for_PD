# Run from the release folder: source("install.R") or Rscript --vanilla install.R
# Install all component dependencies, including the CPU screening environment.
local({
  old_options = options(timeout = max(600, getOption("timeout", 60)))
  on.exit(options(old_options), add = TRUE)

  lib = .libPaths()[1L]
  if (file.access(lib, 2L) != 0L) {
    lib = strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep, fixed = TRUE)[[1L]][1L]
    if (is.na(lib) || !nzchar(lib)) stop("Set R_LIBS_USER to a writable R library directory.")
    lib = path.expand(lib)
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(lib, .libPaths()))
  }
  installed = function(package) nzchar(system.file(package = package))
  cran = c("data.table", "igraph", "Matrix", "reticulate", "BiocManager",
            "rmarkdown", "knitr", "jsonlite")
  bioc = c("edgeR", "rhdf5", "AnnotationDbi", "org.Hs.eg.db")
  missing = cran[!vapply(cran, installed, logical(1))]
  if (installed("reticulate") && packageVersion("reticulate") < "1.43.0") {
    missing = union(missing, "reticulate")
  }
  message("1/3: R dependencies")
  if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org", lib = lib)
  if (!requireNamespace("BiocManager", quietly = TRUE)) stop("BiocManager installation failed.")
  missing = bioc[!vapply(bioc, installed, logical(1))]
  if (length(missing)) BiocManager::install(missing, ask = FALSE, update = FALSE, lib = lib)
  for (package in c(cran, bioc)) {
    if (!requireNamespace(package, quietly = TRUE)) stop("Installation failed for ", package)
    message("  ", package, " ", packageVersion(package))
  }
  if (packageVersion("reticulate") < "1.43.0") stop("reticulate >= 1.43.0 is required; restart R after upgrading.")

  message("2/3: Local Python 3.11 environment")
  env = ".venv"
  python = if (.Platform$OS.type == "windows") ".venv/Scripts/python.exe" else ".venv/bin/python"
  if (!file.exists(python)) {
    if (dir.exists(env)) stop("Incomplete environment at ", env, "; rename it and rerun the installer.")
    # reticulate downloads Python when necessary; virtualenv creates the local env.
    status = reticulate::uv_run_tool("virtualenv", shQuote(env), python_version = "3.11")
    if (status != 0L || !file.exists(python)) stop("Could not create the Python environment.")
  }
  run_python = function(args) {
    status = system2(python, shQuote(args))
    if (status != 0L) stop("Python setup failed; see the command output above.")
  }
  run_python(c("-c", "import sys; assert sys.version_info[:2] == (3, 11), 'DIMER requires Python 3.11 in .venv'"))

  message("3/3: CPU screening dependencies")
  # macOS wheels support CPU directly; Linux/Windows use the explicit CPU build.
  torch_args = if (Sys.info()[["sysname"]] == "Darwin") "torch==2.5.1" else
    c("torch==2.5.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu")
  run_python(c("-m", "pip", "install", "--disable-pip-version-check", torch_args))
  run_python(c("-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"))
  run_python(c("-m", "pip", "check"))
  run_python(c("-c", "import torch, numpy, pandas, scipy, rdkit, gensim, sklearn; assert torch.version.cuda is None; print('CPU screening environment ready.')"))
  if (!rmarkdown::pandoc_available()) {
    message("For HTML rendering, use RStudio (which includes Pandoc) or install Pandoc.")
  }
  message("Installation complete. Open tutorial/DIMER_workflow.Rmd to run the CPU workflow.")
})
