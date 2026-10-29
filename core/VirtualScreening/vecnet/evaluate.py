"""
A5 -- the evaluation harness: reusable primitives for a held-out enrichment claim.

The component has no held-out metric today. Top-100 overlap between the baseline and the
fine-tuned model says the ranking moved; it says nothing about whether it moved
towards the actives. Everything downstream of this module -- every "adaptation
helped" or "did not degrade" sentence -- needs a number that a reviewer can
reproduce, and that number needs three things this file supplies:

  1. **A scaffold split, never a random one.** Chemical libraries are analogue
     series: a random split puts a compound in train and its close analogue in test,
     and the resulting metric mostly measures memorisation. Grouping by RDKit Murcko
     scaffold, so no ring system appears on both sides, is the split that survives
     review.

  2. **Property-matched decoys.** Enrichment against random library compounds
     measures whether the model can tell a drug-like molecule from an arbitrary one,
     which it can, and which is not the question. Decoys matched to the actives on
     size, lipophilicity, polarity and flexibility -- but topologically dissimilar --
     force the metric to be about recognition rather than about bulk properties.
     This matters more here than in most settings: the drug encoder's magnitude is
     essentially molecular size, so unmatched decoys hand the model the answer.

  3. **Early-recognition metrics.** A screen is used by taking the top few hundred
     of half a million. ROC AUC weights the whole ranking equally and can look
     respectable while the usable top of the list is empty; EF at small fractions
     and BEDROC are what correspond to how the output is actually consumed.

Every function also accepts `size_corrected=True`, which ranks on the heavy-atom
residual from `calibrate.py` instead of the raw score. Reporting both is the point:
enrichment that vanishes under size correction is the size artifact, and that is a
result rather than a failure -- it is the same control that overturned the docking
claim.

Scope. These are primitives. The analysis driver that assembles a specific claim --
which actives, which library, which targets -- lives in `analysis/`, per the plan's
implementation map; this module is what it calls, so the split and the metric are
defined once rather than once per script.

numpy + RDKit only. Deliberately no torch and no sklearn: an evaluation must be
runnable on a saved score table without the model that produced it.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .calibrate import average_ranks, heavy_atom_counts, size_residual, spearman

# DUD-E's matching axes, minus formal charge: size, lipophilicity, flexibility,
# H-bonding and polarity. These are the properties a decoy must share with an
# active for the comparison to be about structure rather than about bulk.
DECOY_PROPS = (
    ("MW", Descriptors.MolWt),
    ("LogP", Descriptors.MolLogP),
    ("RotBonds", Descriptors.NumRotatableBonds),
    ("HBA", Descriptors.NumHAcceptors),
    ("HBD", Descriptors.NumHDonors),
    ("TPSA", Descriptors.TPSA),
)

DEFAULT_EF_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
DEFAULT_ALPHA = 20.0        # BEDROC: ~80% of the score from the top 8%
DEFAULT_MAX_TANIMOTO = 0.35
ECFP_BITS = 2048
ECFP_RADIUS = 2


def _mol(smiles):
    """Shared with trainset.parse_molecule in spirit: a 0-atom molecule is not a
    molecule, and must not reach a descriptor or a scaffold."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    return None if (m is None or m.GetNumAtoms() == 0) else m


# ---- 1. scaffold split ---------------------------------------------------
def murcko_scaffold(smiles, include_chirality=False, generic=False):
    """SMILES -> Murcko scaffold SMILES, "" for an acyclic compound, None if unparseable.

    `generic=True` additionally erases atom and bond types, so a pyridine and a
    benzene analogue share a scaffold. Stricter, and the right choice when the
    concern is that a series was split on a single heteroatom swap.
    """
    mol = _mol(smiles)
    if mol is None:
        return None
    try:
        if not generic:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,
                                                       includeChirality=include_chirality)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:
        return None


def scaffold_groups(smiles, include_chirality=False, generic=False, acyclic="group"):
    """SMILES -> {scaffold: [indices]}, in first-appearance order.

    acyclic="group"    : all acyclic compounds share one group. Matches the common
                         reference implementations, so numbers stay comparable.
    acyclic="separate" : each acyclic compound is its own group. Arguably more
                         faithful -- two acyclic molecules share no ring system, so
                         there is no leakage to prevent -- but it is a deviation, so
                         it is not the default.

    Unparseable SMILES get their own singleton groups: they cannot leak, and
    silently dropping them would change the split's compound count.
    """
    groups = {}
    for i, smi in enumerate(smiles):
        sc = murcko_scaffold(smi, include_chirality=include_chirality, generic=generic)
        if sc is None:
            key = "unparseable:%d" % i
        elif sc == "" and acyclic == "separate":
            key = "acyclic:%d" % i
        elif sc == "":
            key = "acyclic"
        else:
            key = sc
        groups.setdefault(key, []).append(i)
    return groups


def scaffold_split(smiles, frac_train=0.8, seed=None, include_chirality=False,
                   generic=False, acyclic="group"):
    """SMILES -> (train_idx, test_idx) with no Murcko scaffold on both sides.

    seed=None : deterministic, largest scaffold group first. This is the standard
        implementation, and it puts the crowded series in train and the singletons
        in test -- a *harder* and more honest test than the reverse.
    seed=int  : the same split rule with a seeded group order, for repeating a claim
        over several splits and reporting the spread. Still a scaffold split: a
        group is never divided.
    """
    smiles = list(smiles)
    n = len(smiles)
    groups = scaffold_groups(smiles, include_chirality=include_chirality,
                             generic=generic, acyclic=acyclic)
    keys = list(groups)
    if seed is None:
        keys.sort(key=lambda k: (-len(groups[k]), k))
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)

    quota = int(round(frac_train * n))
    train, test = [], []
    for k in keys:
        g = groups[k]
        (train if len(train) + len(g) <= quota else test).extend(g)
    return np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(test, dtype=np.int64))


# ---- 2. property-matched decoys -----------------------------------------
def _prop_matrix(smiles):
    """SMILES -> (n, len(DECOY_PROPS)) descriptors, nan rows where unparseable."""
    X = np.full((len(smiles), len(DECOY_PROPS)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles):
        mol = _mol(smi)
        if mol is None:
            continue
        try:
            X[i] = [float(fn(mol)) for _, fn in DECOY_PROPS]
        except Exception:
            pass
    return X


def _fingerprints(smiles, n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = []
    for smi in smiles:
        mol = _mol(smi)
        out.append(None if mol is None else gen.GetFingerprint(mol))
    return out, gen


def property_matched_decoys(active_smiles, candidate_smiles, n_per_active=50,
                            exclude_scaffolds=True, max_tanimoto=DEFAULT_MAX_TANIMOTO,
                            max_candidates=200000, seed=0):
    """Choose decoys from `candidate_smiles` that look like `active_smiles` on bulk
    properties but are structurally unrelated to them.

    Returns (decoy_indices into candidate_smiles, report).

    The two exclusions are what make the metric mean anything:
      - a candidate sharing an active's Murcko scaffold is not a decoy, it is an
        untested analogue, and counting it as a negative penalises the model for
        being right;
      - a candidate with ECFP4 Tanimoto above `max_tanimoto` to any active is the
        same problem without the scaffold match.

    Matching is nearest-neighbour in z-scored property space, without replacement,
    with the actives visited in a seeded random order so the compounds that exhaust
    a crowded region of property space are not always the same ones.

    Cost note: with `exclude_scaffolds=True` every pooled candidate is parsed and
    scaffolded, which is the slow step -- keep `max_candidates` at the size the quota
    actually needs rather than at the library's size.

    `max_candidates` caps the pool by seeded subsampling. The distance computation
    is O(n_actives x n_candidates); a 0.5M library needs no more than a couple of
    hundred thousand candidates to supply fifty decoys per active, and the cap is
    what keeps this from becoming the slowest step in the harness.
    """
    active_smiles = list(active_smiles)
    candidate_smiles = list(candidate_smiles)
    rng = np.random.default_rng(seed)

    pool = np.arange(len(candidate_smiles))
    if max_candidates and len(pool) > max_candidates:
        pool = np.sort(rng.choice(pool, size=int(max_candidates), replace=False))

    A = _prop_matrix(active_smiles)
    C = _prop_matrix([candidate_smiles[i] for i in pool])
    a_ok = np.isfinite(A).all(axis=1)
    c_ok = np.isfinite(C).all(axis=1)
    if not a_ok.any():
        raise ValueError("no parseable active compounds")

    # z-score on the candidate pool: the pool defines the scale the matching is
    # judged on, and using the actives' own spread would make the matching tighter
    # for the actives that happen to be homogeneous.
    mu = np.nanmean(C[c_ok], axis=0)
    sd = np.nanstd(C[c_ok], axis=0)
    sd[sd == 0] = 1.0
    Az = (A - mu) / sd
    Cz = (C - mu) / sd

    banned = set()
    if exclude_scaffolds:
        act_scaffolds = {murcko_scaffold(s) for s in active_smiles}
        act_scaffolds.discard(None)
        act_scaffolds.discard("")
        for j, i in enumerate(pool):
            if murcko_scaffold(candidate_smiles[i]) in act_scaffolds:
                banned.add(j)
    act_canon = set()
    for s in active_smiles:
        m = _mol(s)
        if m is not None:
            act_canon.add(Chem.MolToSmiles(m))

    act_fps = None
    if max_tanimoto is not None:
        act_fps, gen = _fingerprints(active_smiles)
        act_fps = [f for f in act_fps if f is not None]

    available = c_ok.copy()
    for j in banned:
        available[j] = False

    chosen, n_rejected_sim, n_short = [], 0, 0
    order = np.flatnonzero(a_ok)
    rng.shuffle(order)
    for ai in order:
        d = np.linalg.norm(Cz - Az[ai], axis=1)
        d[~available] = np.inf
        cand_order = np.argsort(d, kind="mergesort")
        taken = 0
        for j in cand_order:
            if taken >= n_per_active or not np.isfinite(d[j]):
                break
            smi = candidate_smiles[pool[j]]
            mol = _mol(smi)
            if mol is None or Chem.MolToSmiles(mol) in act_canon:
                available[j] = False
                continue
            if act_fps:
                fp = gen.GetFingerprint(mol)
                if max(DataStructs.BulkTanimotoSimilarity(fp, act_fps)) > max_tanimoto:
                    available[j] = False       # too similar: an analogue, not a decoy
                    n_rejected_sim += 1
                    continue
            chosen.append(int(pool[j]))
            available[j] = False
            taken += 1
        if taken < n_per_active:
            n_short += 1

    report = {
        "n_actives": int(a_ok.sum()),
        "n_candidates_pool": int(len(pool)),
        "n_decoys": len(chosen),
        "decoys_per_active_requested": int(n_per_active),
        "n_actives_short_of_quota": int(n_short),
        "n_rejected_similar": int(n_rejected_sim),
        "n_banned_scaffold": int(len(banned)),
        "max_tanimoto": max_tanimoto,
    }
    return np.asarray(sorted(chosen), dtype=np.int64), report


# ---- 3. early-recognition metrics ---------------------------------------
def _clean(scores, labels):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size != y.size:
        raise ValueError("scores (%d) and labels (%d) differ in length" % (s.size, y.size))
    ok = np.isfinite(s) & np.isfinite(y)
    return s[ok], (y[ok] > 0.5).astype(np.int64)


def _descending_order(scores, tie_break="random", seed=0):
    """Indices of `scores`, best first.

    Ties are broken at random by default, with a seed, because in a decoy benchmark
    the actives are usually contiguous in the input and a stable sort would then
    hand them the whole tie block -- an EF that is an artifact of row order. Pass
    tie_break="stable" when the input order is meaningful.
    """
    s = np.asarray(scores, dtype=np.float64)
    if tie_break == "stable":
        return np.argsort(-s, kind="mergesort")
    if tie_break != "random":
        raise ValueError("tie_break must be 'random' or 'stable'")
    rng = np.random.default_rng(seed)
    jitter = rng.permutation(s.size)
    return np.lexsort((jitter, -s))


def roc_auc(scores, labels):
    """Rank-based ROC AUC with ties averaged; nan when a class is missing."""
    s, y = _clean(scores, labels)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = average_ranks(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def enrichment_factor(scores, labels, fraction=0.01, tie_break="random", seed=0):
    """EF at the top `fraction`: hit rate in the top slice / hit rate overall.

    EF = 1 is no better than picking at random; the theoretical maximum is
    1/fraction and is also reported by `enrichment_report` as `ef_max`, because an
    EF of 40 means something quite different at 1% (max 100) and at 5% (max 20).
    """
    s, y = _clean(scores, labels)
    n, n_pos = s.size, int(y.sum())
    k = int(np.ceil(fraction * n))
    if n == 0 or n_pos == 0 or k == 0:
        return np.nan
    top = _descending_order(s, tie_break=tie_break, seed=seed)[:k]
    return float((y[top].sum() / k) / (n_pos / n))


def bedroc(scores, labels, alpha=DEFAULT_ALPHA, tie_break="random", seed=0):
    """BEDROC (Truchon & Bayly 2007): exponentially weighted early recognition.

    alpha=20 puts ~80% of the weight on the top 8% of the ranking. Bounded in [0, 1]
    with 0.5 the random expectation only in the limit of a small active fraction --
    it is saturating and sensitive to the active/decoy ratio, so it is comparable
    across methods on one dataset, not across datasets.
    """
    s, y = _clean(scores, labels)
    n, n_pos = s.size, int(y.sum())
    if n == 0 or n_pos == 0 or n_pos == n:
        return np.nan
    order = _descending_order(s, tie_break=tie_break, seed=seed)
    ranks = np.flatnonzero(y[order] == 1) + 1.0          # 1-based ranks of actives
    ra = n_pos / float(n)
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rie_num = np.exp(-alpha * ranks / n).sum()
    rie_den = ra * (1.0 - np.exp(-alpha)) / (np.expm1(alpha / n))
    if rie_den == 0:
        return np.nan
    rie = rie_num / rie_den
    scale = ra * np.sinh(alpha / 2.0) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra))
    return float(rie * scale + 1.0 / (1.0 - np.exp(alpha * (1.0 - ra))))


def enrichment_report(scores, labels, smiles=None, heavy_atoms=None,
                      fractions=DEFAULT_EF_FRACTIONS, alpha=DEFAULT_ALPHA,
                      tie_break="random", seed=0, size_corrected=False,
                      residual_method="linear"):
    """One target's ranking -> the metrics, as a flat dict.

    size_corrected=True ranks on the heavy-atom residual instead of the raw score.
    Run it both ways and report both: if the enrichment survives size correction it
    is about the chemistry, and if it does not, that is the finding.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    used = s
    size_rho = np.nan
    if smiles is not None or heavy_atoms is not None:
        if heavy_atoms is None:
            heavy_atoms = heavy_atom_counts(smiles)
        heavy_atoms = np.asarray(heavy_atoms, dtype=np.float64).reshape(-1)
        size_rho = spearman(s, heavy_atoms)
        if size_corrected:
            used, _ = size_residual(s, heavy_atoms, method=residual_method)
    elif size_corrected:
        raise ValueError("size_corrected=True needs `smiles` or `heavy_atoms`")

    ok = np.isfinite(used) & np.isfinite(y)
    out = {
        "n": int(ok.sum()),
        "n_actives": int((y[ok] > 0.5).sum()),
        "n_unscored": int((~np.isfinite(used)).sum()),
        "size_corrected": bool(size_corrected),
        "spearman_size": None if not np.isfinite(size_rho) else round(float(size_rho), 4),
        "roc_auc": _r(roc_auc(used, y)),
        "bedroc_alpha": float(alpha),
        "bedroc": _r(bedroc(used, y, alpha=alpha, tie_break=tie_break, seed=seed)),
    }
    for f in fractions:
        out["ef_%g%%" % (100 * f)] = _r(
            enrichment_factor(used, y, fraction=f, tie_break=tie_break, seed=seed))
        out["ef_max_%g%%" % (100 * f)] = round(1.0 / f, 1)
    return out


def per_target_enrichment(scores, labels, target_names=None, **kwargs):
    """(n_compounds, n_targets) scores + labels -> one report row per target.

    `labels` may be a single (n_compounds,) vector, when the same actives are used
    for every target, or a matrix of the same shape as `scores` when each target has
    its own actives -- which is the tier-1 shape.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    L = np.asarray(labels, dtype=np.float64)
    if L.ndim == 1:
        L = np.repeat(L.reshape(-1, 1), S.shape[1], axis=1)
    if L.shape != S.shape:
        raise ValueError("labels %s do not match scores %s" % (L.shape, S.shape))

    rows = []
    for j in range(S.shape[1]):
        row = enrichment_report(S[:, j], L[:, j], **kwargs)
        row["target"] = (target_names[j] if target_names is not None
                         else "target_%d" % (j + 1))
        rows.append(row)
    return rows


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)
