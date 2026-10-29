"""
One-time asset build: make the labeled DTI fold splits self-contained.

The shipped splits (`data/model/vecnet_unseen_nodes/train_{0..4}.csv`) carry
`InChiKey, target_aa_code, Y` but no SMILES, which forced every caller to supply a
drug SMILES lookup at fit time. `bake_folds` joins the SMILES on once and writes
new splits carrying a `SMILE` column, so the fine-tuner needs no lookup argument.

Why the small lookup and not the ~1 GB chemicals table
-----------------------------------------------------
`data/model/train_inchikey_smiles.csv` (6799 drugs) already resolves 5642 of the
5650 unique drugs in the five training splits. The remaining 8 InChIKeys are
**absent from `data/archived/raw_chemicals/chemicals_01.csv` as well** (verified by
a full scan), so joining against the big table recovers exactly zero extra rows --
the small file is already the maximal join. The 8 unresolvable keys are salts /
metal complexes (charge-suffixed InChIKeys); their rows are dropped here, which is
the same set the previous code silently skipped at fit time.

Import-safe: functions only, no side effects at import.
"""
import os

import pandas as pd

N_FOLDS = 5
FOLD_COLUMNS = ["InChiKey", "target_aa_code", "Y", "SMILE"]


def bake_folds(source_dir, smiles_csv, out_dir, fold_csv_template="train_%d.csv",
               key_col="Label", smiles_col="SMILE"):
    """Join SMILES onto each fold split and write self-contained copies.

    Reads `source_dir/train_%d.csv` + `smiles_csv`, writes `out_dir/train_%d.csv`
    with columns InChiKey, target_aa_code, Y, SMILE. Never writes into source_dir.

    Returns a list of per-fold report dicts (rows in/out, drugs dropped).
    """
    source_dir = str(source_dir).rstrip("/\\")
    out_dir = str(out_dir).rstrip("/\\")
    os.makedirs(out_dir, exist_ok=True)

    lookup = pd.read_csv(smiles_csv, usecols=[key_col, smiles_col])
    lookup = lookup.dropna(subset=[smiles_col]).drop_duplicates(subset=[key_col])
    mapping = dict(zip(lookup[key_col].values, lookup[smiles_col].values))

    report = []
    for r in range(N_FOLDS):
        src = "%s/%s" % (source_dir, fold_csv_template % r)
        df = pd.read_csv(src, usecols=["InChiKey", "target_aa_code", "Y"])
        df["SMILE"] = df["InChiKey"].map(mapping)

        n_in = len(df)
        dropped_keys = sorted(set(df.loc[df["SMILE"].isna(), "InChiKey"]))
        out = df.dropna(subset=["SMILE"])[FOLD_COLUMNS]
        out.to_csv("%s/%s" % (out_dir, fold_csv_template % r), index=False)

        report.append({
            "fold": r,
            "rows_in": n_in,
            "rows_out": len(out),
            "rows_dropped": n_in - len(out),
            "drugs_unresolved": len(dropped_keys),
            "unresolved_keys": dropped_keys,
            "positives": int(out["Y"].sum()),
        })
    return report
