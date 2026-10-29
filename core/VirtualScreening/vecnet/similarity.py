"""The optional ligand-similarity channel.

What it is. The sequence model scores a compound from the protein sequence alone.
When the caller also knows ONE compound that binds the target -- a co-crystallised
ligand, or any published inhibitor -- that is extra information the sequence model
cannot use. This module turns it into a second ranking and merges the two.

What it needs. A known binder, not a protein structure. A SMILES string is enough;
a mol2/sdf path works too and is parsed to the same thing. The channel is therefore
ligand-based: it is available whenever a binder is known, with or without a structure.

How the merge works, and why it cannot damage the top of the list.

    combined = round(tanimoto / w) * w  +  sequence_percentile * w * 0.999

The first term bins similarity at resolution `w`. The second is strictly smaller
than one bin (0.999 * w < w), so it can never lift a compound out of its bin. Bin
order is therefore decided by similarity alone and the sequence model only orders
compounds *within* a bin -- i.e. only where similarity cannot tell them apart.

`w` is the resolution below which two Tanimoto values are treated as
indistinguishable. Its limits are the two pure rankings: w -> 0 is similarity only,
w -> 1 is sequence only. The default 0.01 is the natural quantum of the measure --
ECFP4 fingerprints of drug-like molecules carry ~50 on-bits, so a one-bit change
moves Tanimoto by ~1/100 -- which is why it barely perturbs the similarity order.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ECFP_RADIUS = 2
ECFP_BITS = 2048
DEFAULT_BAND = 0.01

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)


def _sanitized(mol):
    """A reference whose aromaticity is not perceived produces a fingerprint in a
    different chemical alphabet from the library's, which silently halves the
    channel's performance. Never fingerprint an unsanitized molecule."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def load_reference(reference):
    """SMILES string | mol2/sdf path | rdkit Mol -> Mol, or None if unusable."""
    if reference is None:
        return None
    if isinstance(reference, Chem.Mol):
        return _sanitized(Chem.Mol(reference))
    text = str(reference).strip()
    if not text:                      # "" parses to a valid EMPTY molecule, not to None
        return None
    low = text.lower()
    if low.endswith(".mol2"):
        m = Chem.MolFromMol2File(text, sanitize=True)
        return m if m is not None else _sanitized(Chem.MolFromMol2File(text, sanitize=False))
    if low.endswith((".sdf", ".mol")):
        supplier = Chem.SDMolSupplier(text, sanitize=True)
        for m in supplier:
            if m is not None:
                return m
        return None
    return _sanitized(Chem.MolFromSmiles(text))   # SMILES: sanitized by default


def load_references(reference):
    """One reference or many -> list of Mol. Unusable entries are dropped.

    A single crystal ligand is the degenerate case. Several known binders are the
    normal one, and they are strictly more informative: a target's chemistry is
    rarely one scaffold, and a library compound only has to resemble ONE of the
    things already known to bind.
    """
    if reference is None:
        return []
    items = ([reference] if isinstance(reference, (str, Chem.Mol))
             else list(reference))
    mols = [load_reference(x) for x in items]
    return [m for m in mols if m is not None]


def similarity_scores(reference, smiles, max_references=None):
    """-> (n,) Tanimoto to the reference set; nan where RDKit cannot parse.

    With several references the score is the MAXIMUM similarity to any of them, so a
    compound resembling one known binder is not penalised for resembling none of the
    others. Cost is linear in the number of references -- 1.82M compounds against 50
    references takes ~12 s, against 3,000 takes ~14 min -- which is what
    `max_references` is for. It keeps the first N, so pass the ones you trust most
    first (the bundled sets are ordered by potency).
    """
    refs = load_references(reference)
    if not refs:
        raise ValueError("no reference ligand could be parsed or sanitized; "
                         "pass a SMILES string if a structure file will not load")
    if max_references is not None and len(refs) > max_references:
        refs = refs[:max_references]
    rfps = [_gen.GetFingerprint(m) for m in refs]

    # Fingerprint the library ONCE, then sweep the references over it. The other way
    # round re-fingerprints the library per reference and is len(refs) times slower.
    fps, idx = [], []
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi) if not isinstance(smi, Chem.Mol) else smi
        if m is None or m.GetNumAtoms() == 0:
            continue
        fps.append(_gen.GetFingerprint(m))
        idx.append(i)
    out = np.full(len(smiles), np.nan, dtype=np.float64)
    if not fps:
        return out
    best = np.zeros(len(fps), dtype=np.float64)
    for rfp in rfps:
        np.maximum(best, np.asarray(DataStructs.BulkTanimotoSimilarity(rfp, fps)),
                   out=best)
    out[np.asarray(idx)] = best
    return out


def _percentile(x):
    """-> (n,) rank in [0,1], high = good. Unscored compounds rank last."""
    x = np.asarray(x, dtype=np.float64)
    v = np.where(np.isfinite(x), x, -np.inf)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=np.float64)
    r[order] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def combine(sequence_scores, similarity, band=DEFAULT_BAND):
    """Similarity-primary merge. See the module docstring for why it is safe.

    band=0 returns similarity alone; band=1 returns the sequence ranking.
    """
    if not 0.0 <= band <= 1.0:
        raise ValueError("band must lie in [0, 1]; got %r" % (band,))
    sim = np.nan_to_num(np.asarray(similarity, dtype=np.float64), nan=0.0)
    if band == 0.0:
        return sim
    seq = _percentile(sequence_scores)
    if band >= 1.0:
        return seq
    return np.round(sim / band) * band + seq * band * 0.999


# ---------------------------------------------------------------------------
# Choosing the band from data.
#
# The band is the one free parameter of the channel, and the right value is a
# property of the target, not of the method: a target whose known binders share a
# chemotype is well served by a fine band, one whose binders are chemically
# scattered is not served at all and should fall back to the sequence model. The
# benchmark keeps the band fixed at the default so that comparison across 117
# proteins involves no tuning; an application picks it per target, here.
#
# The calibration set is a set of known binders held apart from whatever the screen
# is asked to nominate -- it decides the parameter, it is not itself the answer.
# w = 1.0 is always in the grid and returns the sequence ranking exactly, so the
# rule below cannot select a band that does worse than the sequence model alone on
# the calibration data. That is the whole safety argument: the channel is free to
# help and is switched off where it does not.
# ---------------------------------------------------------------------------

DEFAULT_BAND_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)


def enrichment_at(scores, labels, fraction=0.01):
    """Enrichment factor at the top `fraction` of the ranking."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n = max(int(round(len(scores) * fraction)), 1)
    top = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")[:n]
    base = labels.sum() / len(labels)
    return float(labels[top].sum()) / n / base if base > 0 else float("nan")


def calibrate_band(sequence_scores, similarity, labels,
                   bands=DEFAULT_BAND_GRID, fraction=0.01):
    """-> (band, [(band, enrichment), ...]) from a labelled calibration set.

    `labels` marks the known binders among the pooled compounds; everything else is
    background. The chosen band maximises enrichment at `fraction`, ties broken
    toward the LARGER band -- i.e. toward the sequence model, so an ambiguous
    calibration leaves the ranking alone rather than perturbing it.
    """
    labels = np.asarray(labels).astype(bool)
    if not labels.any():
        raise ValueError("calibration set contains no known binders")
    if labels.all():
        raise ValueError("calibration set contains no background compounds")
    grid = sorted(set(float(b) for b in bands) | {1.0})   # 1.0 is the fallback
    table = [(w, enrichment_at(combine(sequence_scores, similarity, band=w),
                               labels, fraction)) for w in grid]
    best = max(e for _, e in table)
    return max(w for w, e in table if e >= best), table
