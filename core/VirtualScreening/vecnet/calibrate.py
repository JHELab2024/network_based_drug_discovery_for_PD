"""
A2 -- calibration: make a VecNet score mean something.

Two corrections, applied in this order, per target:

  1. **Library percentile.** A raw P(bind) is not comparable across targets. The
     target vector enters the network before the head and shifts the entire score
     distribution, so one target's 0.62 can sit at the library's 3rd percentile and
     another's at its 97th. Any cross-target statement -- "these are the twenty best
     compound/target pairs", a consensus figure, a ranked nomination list -- is
     meaningless until each score is expressed against its own target's library
     distribution. The reference distribution is the *user's library itself*, which
     is the only distribution the screen actually samples.

  2. **Heavy-atom residualisation.** The Mol2vec drug vector is a sum over Morgan
     substructure identifiers, so its magnitude is essentially molecular size; the
     score therefore correlates with heavy-atom count, and the sign of that
     correlation is set by the target rather than by the chemistry. Ranking on the
     raw score then ranks on size, which is what produced fragment-sized top-500
     lists for some targets and drug-sized ones for others. This is the same
     correction that overturned the docking enrichment claim, applied prospectively
     rather than as a post-mortem.

Which quantity gets residualised
--------------------------------
The residual is taken on the **percentile**, not the raw score. Residualising the
raw score leaves a per-target scale -- exactly what step 1 exists to remove -- so
the two corrections would not compose. Pass `residual_of="score"` for the
alternative; the plan does not settle it and this is the reading that keeps both
properties.

What this module is not
-----------------------
Not a probability calibration: nothing here claims the output is a better estimate
of P(bind). It makes scores *comparable* (across targets) and *decorrelated* from a
known artifact (size). A calibrated score that is still uninformative is still
uninformative -- that is what A3's gate and A5's enrichment harness are for.

numpy + RDKit only. No torch, no sklearn: calibration must be runnable on a saved
score table, long after the screen, on a machine that has neither.
"""
import numpy as np
from rdkit import Chem

DEFAULT_METHOD = "linear"
DEFAULT_DEGREE = 1
DEFAULT_BINS = 20


# ---- molecular size ------------------------------------------------------
def heavy_atom_counts(smiles):
    """SMILES -> heavy-atom count per compound, nan where RDKit cannot parse it.

    Unique SMILES are parsed once: a screening library is full of duplicates and
    this is called once per run, not once per target.
    """
    smiles = list(smiles)
    unique = list(dict.fromkeys(smiles))
    got = {}
    for s in unique:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() else None
        # A 0-atom molecule (RDKit's answer for "") is not size 0, it is not a
        # molecule; nan keeps it out of the regression instead of anchoring it.
        got[s] = float(mol.GetNumHeavyAtoms()) if (mol is not None and mol.GetNumAtoms()) else np.nan
    return np.fromiter((got[s] for s in smiles), dtype=np.float64, count=len(smiles))


# ---- ranking primitives --------------------------------------------------
def average_ranks(v):
    """1-based ranks with ties averaged, vectorised. Input must be finite.

    Public because `evaluate.py` needs exactly this tie handling: a metric and a
    percentile that disagree about ties would make a calibrated ranking and its
    enrichment inconsistent.
    """
    n = v.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    new = np.ones(n, dtype=bool)
    new[1:] = sv[1:] != sv[:-1]
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    avg = starts + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[grp]
    return ranks


def library_percentile(scores):
    """Scores -> percentile in (0, 100) against the library's own distribution.

    Ties share an averaged rank. The half-rank offset -- 100*(rank-0.5)/n rather
    than 100*rank/n -- keeps the best compound off exactly 100 and the worst off
    exactly 0, so a percentile can be transformed (logit, z) without a special case
    for the endpoints. nan in, nan out, and nan compounds are excluded from n.
    """
    s = np.asarray(scores, dtype=np.float64)
    out = np.full(s.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(s)
    n = int(ok.sum())
    if n == 0:
        return out
    out[ok] = 100.0 * (average_ranks(s[ok]) - 0.5) / n
    return out


def spearman(x, y):
    """Spearman rho over the pairs where both are finite; nan if undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx, ry = average_ranks(x[ok]), average_ranks(y[ok])
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


# ---- the size correction -------------------------------------------------
def size_residual(values, heavy_atoms, method=DEFAULT_METHOD, degree=DEFAULT_DEGREE,
                  n_bins=DEFAULT_BINS):
    """values - E[values | heavy-atom count]  ->  (residual, fit report).

    method="linear" : ordinary least squares of `values` on heavy-atom count, of
        `degree` (1 by default). Cheap, and the size effect is close to monotone
        over the range a screening library covers.
    method="binned" : subtract the mean within quantile bins of heavy-atom count.
        Assumes nothing about the shape, at the cost of `n_bins` degrees of freedom;
        this is the form the docking size control used.

    A compound with a positive residual scores better than its size alone predicts,
    which is the only sense in which the model has said something about *it*. nan
    propagates: a compound with no parseable structure gets no residual rather than
    a residual of zero, which would place it at the middle of the ranking.
    """
    v = np.asarray(values, dtype=np.float64)
    h = np.asarray(heavy_atoms, dtype=np.float64)
    if v.shape != h.shape:
        raise ValueError("values %s and heavy_atoms %s must have the same shape"
                         % (v.shape, h.shape))
    resid = np.full(v.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(h)
    report = {"method": method, "n_used": int(ok.sum()),
              "spearman_size": spearman(v, h)}
    if ok.sum() < 3 or np.unique(h[ok]).size < 2:
        # Nothing to regress against: hand back the centred values so downstream
        # ranking still works, and say so in the report rather than silently.
        resid[ok] = v[ok] - v[ok].mean() if ok.any() else np.nan
        report["note"] = "too few distinct heavy-atom counts; residual is centred value"
        return resid, report

    if method == "linear":
        coef = np.polyfit(h[ok], v[ok], int(degree))
        fitted = np.polyval(coef, h[ok])
        report["coefficients"] = [float(c) for c in coef]
        report["degree"] = int(degree)
    elif method == "binned":
        edges = np.unique(np.quantile(h[ok], np.linspace(0, 1, int(n_bins) + 1)))
        idx = np.clip(np.searchsorted(edges, h[ok], side="right") - 1,
                      0, max(len(edges) - 2, 0))
        sums = np.bincount(idx, weights=v[ok], minlength=len(edges) - 1)
        cnts = np.bincount(idx, minlength=len(edges) - 1)
        means = np.divide(sums, cnts, out=np.full(len(cnts), np.nan), where=cnts > 0)
        fitted = means[idx]
        report["n_bins"] = int(len(cnts))
    else:
        raise ValueError("unknown method %r (expected 'linear' or 'binned')" % (method,))

    resid[ok] = v[ok] - fitted
    ss_tot = float(np.var(v[ok])) * ok.sum()
    ss_res = float(np.nansum(np.square(v[ok] - fitted)))
    report["r2_size"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return resid, report


# ---- the public entry points --------------------------------------------
def calibrate_scores(scores, smiles=None, heavy_atoms=None, method=DEFAULT_METHOD,
                     degree=DEFAULT_DEGREE, n_bins=DEFAULT_BINS,
                     residual_of="percentile"):
    """One target's scores -> both calibrated columns.

    Returns a dict with
      percentile      : library percentile of the raw score (comparable across targets)
      size_residual   : that percentile minus what heavy-atom count predicts
      residual_rank   : 1 = highest residual, i.e. the ranking to actually use
      report          : the size-regression fit and its size correlation

    Give it `smiles` or a precomputed `heavy_atoms`; the latter avoids re-parsing
    the library once per target.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if heavy_atoms is None:
        if smiles is None:
            raise ValueError("pass `smiles` or a precomputed `heavy_atoms`")
        heavy_atoms = heavy_atom_counts(smiles)
    h = np.asarray(heavy_atoms, dtype=np.float64).reshape(-1)
    if h.size != s.size:
        raise ValueError("scores (%d) and heavy_atoms (%d) differ in length"
                         % (s.size, h.size))

    if residual_of not in ("percentile", "score"):
        raise ValueError("residual_of must be 'percentile' or 'score'")
    pct = library_percentile(s)
    base = pct if residual_of == "percentile" else s
    resid, report = size_residual(base, h, method=method, degree=degree, n_bins=n_bins)

    rank = np.full(resid.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(resid)
    if ok.any():
        rank[ok] = (int(ok.sum()) + 1) - average_ranks(resid[ok])   # 1 = best

    report.update({
        "residual_of": residual_of,
        "n_scored": int(np.isfinite(s).sum()),
        "mean_score": float(np.nanmean(s)) if np.isfinite(s).any() else np.nan,
    })
    return {"percentile": pct, "size_residual": resid, "residual_rank": rank,
            "report": report}


def calibrate_matrix(scores, smiles=None, heavy_atoms=None, target_names=None,
                     method=DEFAULT_METHOD, degree=DEFAULT_DEGREE,
                     n_bins=DEFAULT_BINS, residual_of="percentile"):
    """(n_compounds, n_targets) scores -> the same two columns, per target.

    Per target because both corrections are per target: the percentile is against
    that target's own library distribution, and the size regression is refitted
    because its *sign* changes from target to target -- that is the failure mode,
    not a nuisance parameter to share.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if heavy_atoms is None:
        if smiles is None:
            raise ValueError("pass `smiles` or a precomputed `heavy_atoms`")
        heavy_atoms = heavy_atom_counts(smiles)
    h = np.asarray(heavy_atoms, dtype=np.float64).reshape(-1)

    pct = np.full(S.shape, np.nan, dtype=np.float64)
    resid = np.full(S.shape, np.nan, dtype=np.float64)
    rank = np.full(S.shape, np.nan, dtype=np.float64)
    reports = []
    for j in range(S.shape[1]):
        out = calibrate_scores(S[:, j], heavy_atoms=h, method=method, degree=degree,
                               n_bins=n_bins, residual_of=residual_of)
        pct[:, j] = out["percentile"]
        resid[:, j] = out["size_residual"]
        rank[:, j] = out["residual_rank"]
        rep = dict(out["report"])
        rep["target"] = (target_names[j] if target_names is not None
                         else "target_%d" % (j + 1))
        reports.append(rep)
    return {"percentile": pct, "size_residual": resid, "residual_rank": rank,
            "heavy_atoms": h, "report": reports}
