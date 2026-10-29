# Reference binders

These files provide target sequences and known binders for VecNet's ligand-similarity
channel. The dataset contains 1,680 human targets and 865,373 target–compound pairs
from ChEMBL 36.

## Selection

The dataset includes human single-protein targets with at least 10 distinct compounds
and measurements meeting both criteria:

- `standard_type` is IC50, Ki, or Kd.
- `pchembl_value` is at least 6, corresponding to a concentration of 1 µM or lower.

The minimum of 10 binders supports automatic calibration of the similarity band.

## Files

- `targets.csv`: target symbol, UniProt and ChEMBL identifiers, protein name,
  sequence, length, and binder count. It also includes PDB identifier, ligand code,
  and reference SMILES for the seven application targets.
- `binders.csv.gz`: target and compound identifiers, SMILES, best pChEMBL value,
  and measurement types. Binders are ordered by pChEMBL when loaded; the
  `max_references` setting limits how many are used.

For targets without a reference ligand, known binders serve as the similarity
references. The similarity band is calibrated for the library being screened.
Users can also supply their own `known_binders`.

The R workflow is shown in [DIMER_workflow.Rmd](../../../../tutorial/DIMER_workflow.Rmd).

## Source

The reference files were prepared from a local ChEMBL 36 SQLite database. ChEMBL is
released under CC BY-SA 3.0. Cite Zdrazil et al., *Nucleic Acids Research* (2024).
