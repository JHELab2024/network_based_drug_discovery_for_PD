"""Bundled reference data for every ChEMBL target with enough known binders to calibrate.

The ligand-similarity channel needs two things the sequence model does not: a known
binder to compare against, and enough further binders to calibrate the band. Both are
public -- a PDB entry and a ChEMBL query -- but assembling them is the step most
likely to stop someone trying the channel at all. So they ship with the package.

    import vecnet
    result = vecnet.screen(library=my_library, **vecnet.reference_binders("LRRK2"))

That call supplies the sequence, the crystal-ligand reference and the known binders
in one go, and the band is then chosen from those binders against `my_library`.

What is NOT shipped is the band itself. The band is a property of the library being
screened, not of the target: the values reported in the paper were calibrated against
a 1.82M purchasable set and would be the wrong number for a different library. The
inputs travel; the fitted parameter does not.
"""
import csv
import gzip
import os

DIRNAME = "reference_binders"


def _dir(assets_root):
    here = os.path.dirname(os.path.abspath(__file__))
    root = assets_root or os.path.dirname(here)
    return os.path.join(root, "assets", DIRNAME)


def reference_targets(assets_root=None):
    """-> list of dicts, one per bundled target: symbol, uniprot, pdb, het_code, ..."""
    path = os.path.join(_dir(assets_root), "targets.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("bundled reference data not found at %s" % path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["length_aa"] = int(r["length_aa"])
        r["n_binders"] = int(r["n_binders"])
    return rows


def known_binders(symbol=None, min_pchembl=None, assets_root=None):
    """-> {symbol: [SMILES, ...]} for one target, or all of them."""
    path = os.path.join(_dir(assets_root), "binders.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError("bundled reference data not found at %s" % path)
    want = None if symbol is None else set(_as_list(symbol))
    rows = {}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if want is not None and r["symbol"] not in want:
                continue
            try:
                p = float(r["pchembl_best"])
            except (TypeError, ValueError):
                continue
            if min_pchembl is not None and p < min_pchembl:
                continue
            rows.setdefault(r["symbol"], []).append((p, r["smiles"]))
    # Most potent first: `max_references` keeps a prefix, so the order is the policy
    # for which binders get used as similarity queries when the set is capped.
    return {k: [smi for _, smi in sorted(v, key=lambda t: -t[0])]
            for k, v in rows.items()}


def reference_binders(symbol=None, min_pchembl=None, assets_root=None):
    """-> dict(targets, reference_ligand, known_binders), ready to splat into screen().

    symbol : one target, a list of them, or None for every bundled target.
    min_pchembl : keep only binders at or above this potency. The default keeps all
        of them, because the band is chosen by enrichment and a weak binder that the
        similarity channel ranks well is still evidence the channel is working.
    """
    rows = reference_targets(assets_root)
    want = None if symbol is None else set(_as_list(symbol))
    if want is not None:
        missing = want - {r["symbol"] for r in rows}
        if missing:
            raise KeyError("no bundled reference data for %s; available: %s"
                           % (", ".join(sorted(missing)),
                              ", ".join(r["symbol"] for r in rows)))
        rows = [r for r in rows if r["symbol"] in want]
    binders = known_binders(symbol, min_pchembl=min_pchembl, assets_root=assets_root)
    return {
        "targets": {r["symbol"]: r["sequence"] for r in rows},
        # Only the paper's seven carry a crystal ligand; a blank must not travel as
        # if it were one.
        "reference_ligand": {r["symbol"]: r["reference_smiles"] for r in rows
                             if r.get("reference_smiles")},
        "known_binders": {r["symbol"]: binders.get(r["symbol"], []) for r in rows},
    }


def _as_list(x):
    return [x] if isinstance(x, str) else list(x)
