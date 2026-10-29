# DIMER

**D**irect and **I**ndirect **M**odule **E**valuation for **R**epurposing and screening.

DIMER prioritizes drugs and drug pairs using network proximity, evaluates
transcriptional reversal with RevERT, and screens compound libraries against
protein targets with VecNet. This pipeline release includes code, pretrained weights,
and a CPU tutorial in R Markdown.
The VecNet predictor is derived from AI-Bind and uses Mol2vec and ProtVec encoders.

```text
model_release/
  core/
    NetworkProximity/
    FunctionalDirection/
    VirtualScreening/       Python code, R binding, and pretrained assets
  data/                     optional full reference data and loader
  tutorial/
    DIMER_workflow.Rmd       installation and the complete CPU workflow
    input/                  example inputs
  install.R
  requirements.txt
```

## 1. Install

### Windows and Mac

1. Install R 4.5 and RStudio. Use 64-bit R on Intel/AMD Windows computers and
   the arm64 version of R on Apple silicon Macs.
2. Extract `DIMER_release.tar.gz` into a folder named `model_release`.
3. Open `tutorial/DIMER_workflow.Rmd` in RStudio.
4. Choose **Session > Set Working Directory > To Source File Location**.
5. Run the installation chunk:

```r
source("../install.R", chdir = TRUE)
```

The installer sets up the R packages and Python CPU dependencies automatically.
Once it finishes, click **Knit** or run the remaining chunks in order.

### Linux

Unpack `DIMER_release.tar.gz` wherever you want to work:

```bash
mkdir model_release
tar -xzf DIMER_release.tar.gz -C model_release
```

With R 4.5 installed, run this from the unpacked release folder in R or RStudio:

```r
source("install.R")
```

The installer adds missing R packages and sets up Python 3.11 with CPU
dependencies in `.venv/`. Internet access is needed for installation. On Linux,
install R build tools (`r-base-dev` on Ubuntu) first. HTML rendering needs RStudio or Pandoc.
DIMER functions are loaded with `source()`. No GPU or CUDA is required.

## 2. Run the R Markdown tutorial

Open [tutorial/DIMER_workflow.Rmd](tutorial/DIMER_workflow.Rmd) in RStudio.
After installation, click **Knit**, or run the chunks with `tutorial/` as your
working directory. To render from the release directory:

```r
rmarkdown::render("tutorial/DIMER_workflow.Rmd")
```

The tutorial uses simulated network and expression data for `DrugA`–`DrugD`,
then screens eight real compound structures against LRRK2. It runs offline
after installation. HTML is saved beside the Rmd; results and software versions
go into `tutorial/output/`.

## Reference data

Full network and CMap reference data are optional; see
[data/README.md](data/README.md).

Third-party data and assets retain their source terms; see the
[core/VirtualScreening/assets/reference_binders/README.md](core/VirtualScreening/assets/reference_binders/README.md).
