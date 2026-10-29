"""
User library -> importance weights over the labeled fold pairs.

What this replaced, and why
---------------------------
The previous version defined a "chemical subspace" as the library's InChIKey set
plus a min/max box over MW, TPSA, HBD, HBA and LogP, and kept a labeled compound
if it fell inside. Over a 544,638-compound library that box is fixed by the five
most extreme compounds per axis, so it is effectively unbounded: it retained
nearly every fold row, `fit()` was therefore ordinary continued training, and the
ranking shift it produced was drift rather than adaptation. Two pathologies, both
structural rather than tuning issues:

  - the box is set by the library's extremes, so it *widens with library size* --
    the same procedure means something different for a 1k and a 500k library;
  - it treats a compound just inside and a compound just outside as wholly
    different, and everything inside as identical.

Both disappear once the procedure is named correctly. The user supplies no labels
and curates nothing; what the library supplies is a description of the
*distribution the model will be deployed on*. That is transfer learning under
covariate shift, and the estimator for it is a density ratio, obtained by training
a classifier to separate the two compound sets:

    w(x) = p(library | x) / (1 - p(library | x))

Each labeled pair then contributes to the loss in proportion to how much its
compound resembles the user's chemistry. No threshold, no envelope, and -- the
point -- no dependence on library size: the classifier sees a fixed-size sample of
the library (`max_library_sample`), so a 500k library and a 50k library describing
the same chemistry yield the same weights.

The box is kept, reachable as `mode="legacy_box"`, because it is the degenerate
case of this estimator with hard 0/1 weights. Keeping it runnable is what makes
that reduction a demonstration rather than an assertion.

Why the odds and not the probability
------------------------------------
With `class_weight="balanced"` the classifier is fitted as though the two sets were
equally likely a priori, so its posterior odds is the density ratio
p_library(x) / p_aibind(x) with the sampling prior already divided out. That is why
the sample size of either side does not enter the weights.

Weight conditioning (three steps, each with a distinct job)
-----------------------------------------------------------
Raw odds are unusable as loss weights: they span orders of magnitude and a single
compound can dominate a fold. `normalise_weights()` does, in order,

  1. divide by the median -- makes the clip threshold scale-free, so "10" means
     "ten times the typical compound" rather than an absolute odds that would move
     with the classifier's calibration;
  2. clip to [1/clip, clip] -- bounds any one pair's leverage, which is what keeps
     the effective sample size from collapsing onto a handful of compounds; the
     variance of an importance-weighted estimator is driven by its largest weights;
  3. rescale to mean 1 -- keeps the loss on the same scale as the unweighted
     baseline, so `lr` does not silently change meaning and the epoch-0 versus
     epoch-k comparison A4 relies on is like for like.

Step 3 is also what makes the legacy box exactly recoverable: hard 0/1 weights with
the zeros dropped rescale to all-ones, i.e. the old unweighted training set.

Invariants respected here
-------------------------
  - the all-zero / unparseable drug guard (`parse_molecule`) is unchanged, and a
    compound that fails it gets weight 0 and is dropped -- an empty molecule can
    never become a training example;
  - the weight function is label-free and shared across the five folds, but the
    conditioning in step 1-3 is computed *within* each fold, so the folds stay
    independent.
"""
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# ---- modes ---------------------------------------------------------------
MODE_DENSITY_RATIO = "density_ratio"    # A1: importance weighting (default)
MODE_LEGACY_BOX = "legacy_box"          # the pre-A1 min/max envelope, kept as the
                                        # degenerate hard-0/1 case of the above
DEFAULT_MODE = MODE_DENSITY_RATIO

# Axes of the legacy box. Kept because `mode="legacy_box"` is kept.
PROPS = ("MW", "TPSA", "HBD", "HBA", "LogP")

# Descriptor block for the domain classifier. Wider than PROPS because those five
# axes were chosen to be human-interpretable, not to describe a distribution: what
# separates a curated library (BBB-permeable, say) from a DTI training set is size,
# polarity, flexibility, aromaticity and saturation together. Cheap 2D descriptors
# only -- nothing here should cost more than the RDKit parse it already needs.
DOMAIN_DESCRIPTORS = (
    ("MW", Descriptors.MolWt),
    ("TPSA", Descriptors.TPSA),
    ("HBD", Descriptors.NumHDonors),
    ("HBA", Descriptors.NumHAcceptors),
    ("LogP", Descriptors.MolLogP),
    ("HeavyAtoms", Descriptors.HeavyAtomCount),
    ("RotBonds", Descriptors.NumRotatableBonds),
    ("Rings", Descriptors.RingCount),
    ("AromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("Heteroatoms", Descriptors.NumHeteroatoms),
    ("MolMR", Descriptors.MolMR),
)

# Defaults. `MAX_LIBRARY_SAMPLE` is the whole point of A1's "no dependence on
# library size": the density estimate saturates long before 20k compounds, so the
# classifier's cost -- and its answer -- stop moving once the library is bigger
# than this.
MAX_LIBRARY_SAMPLE = 20000
WEIGHT_CLIP = 10.0
ECFP_BITS = 1024
ECFP_RADIUS = 2
_PROB_EPS = 1e-6      # keeps a perfectly separated compound's odds finite


# ---- molecule handling (unchanged; invariant 4 lives here) ---------------
def parse_molecule(smiles):
    """SMILES -> RDKit mol, or None if it is not a molecule.

    MolFromSmiles("") returns an empty **0-atom molecule**, not None. Left
    unguarded that phantom would enter the descriptor matrix with MW=TPSA=LogP=0,
    drag every lower bound to zero in the legacy box and sit at the extreme of the
    density estimate in the new one. A blank cell in a user's library CSV is all it
    takes. Downstream, a molecule that fails here gets weight 0 and is dropped, so
    it never becomes a training example.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def descriptors(mol):
    """The five legacy box axes, in PROPS order. Unchanged from the pre-A1 version."""
    return (
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        float(Descriptors.NumHDonors(mol)),
        float(Descriptors.NumHAcceptors(mol)),
        Descriptors.MolLogP(mol),
    )


def domain_descriptors(mol):
    """The domain-classifier descriptor row, or None if any descriptor blows up.

    A non-finite descriptor (a disconnected metal complex can produce one) would
    poison the scaler and every weight with it, so such a compound is dropped from
    the *classifier's* training data rather than being imputed.
    """
    try:
        row = [float(fn(mol)) for _, fn in DOMAIN_DESCRIPTORS]
    except Exception:
        return None
    arr = np.asarray(row, dtype=np.float64)
    return arr if np.isfinite(arr).all() else None


def _ecfp_generator(n_bits=ECFP_BITS, radius=ECFP_RADIUS):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _sample_indices(n, k, seed):
    """A seeded, order-preserving subsample of range(n). Sorted so the RDKit parse
    walks the library in its original order (better locality, reproducible logs)."""
    if k is None or n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(k), replace=False))


# ---- weight conditioning -------------------------------------------------
def normalise_weights(raw, clip=WEIGHT_CLIP):
    """Raw odds (nan allowed) -> loss weights with mean 1 over the survivors.

    See the module docstring for why this is median-normalise, then clip, then
    mean-normalise. `nan` and non-positive inputs come back as 0.0, which is the
    signal to drop that pair: it is either unparseable (invariant 4) or outside the
    legacy box.

    Returns (weights, kept_mask).
    """
    w = np.asarray(raw, dtype=np.float64).copy()
    bad = ~np.isfinite(w) | (w <= 0.0)
    w[bad] = 0.0
    kept = ~bad
    if not kept.any():
        return w, kept

    med = float(np.median(w[kept]))
    if med > 0.0:
        w[kept] /= med                       # 1: scale-free clip threshold
    if clip and clip > 0:
        lo, hi = 1.0 / float(clip), float(clip)
        w[kept] = np.clip(w[kept], lo, hi)   # 2: bound single-pair leverage
    mean = float(w[kept].mean())
    if mean > 0.0:
        w[kept] /= mean                      # 3: baseline loss scale preserved
    return w, kept


def effective_sample_size(weights):
    """(sum w)^2 / sum w^2 -- how many pairs the weighted fit effectively had.

    Equal weights give back the pair count exactly, which is why the legacy box's
    ESS is its retained-pair count and the density-ratio ESS is always lower: that
    gap is the price of honesty about the shift, and it is A3's second diagnostic.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    s = w.sum()
    if s <= 0:
        return 0.0
    return float(s * s / np.square(w).sum())


# ---- A3: how to read the domain-classifier AUC ---------------------------




# ---- the subspace object -------------------------------------------------
class ChemicalSubspace:
    """What the user's library says about the deployment distribution.

    In `density_ratio` mode this wraps a fitted domain classifier and turns SMILES
    into importance weights. In `legacy_box` mode it wraps the old InChIKey set plus
    min/max envelope and returns hard 0/1 weights -- the same object, the degenerate
    case.

    Unpacks as the pre-A1 3-tuple, so `keys, envelope, key2smi = build_subspace(...)`
    still works.
    """

    def __init__(self, mode, keys, envelope, key2smi, pipeline=None,
                 featurise=None, domain_auc=None, n_library=0, n_background=0,
                 envelope_from_sample=False, weight_clip=WEIGHT_CLIP,
                 classifier=None, features=None):
        self.mode = mode
        self.keys = keys
        self.envelope = envelope
        self.key2smi = key2smi
        self.pipeline = pipeline
        self._featurise = featurise
        self.domain_auc = domain_auc
        self.n_library = int(n_library)
        self.n_background = int(n_background)
        self.envelope_from_sample = bool(envelope_from_sample)
        self.weight_clip = weight_clip
        self.classifier = classifier
        self.features = features

    # backward compatibility: the pre-A1 return contract
    def __iter__(self):
        return iter((self.keys, self.envelope, self.key2smi))

    def __repr__(self):
        return ("ChemicalSubspace(mode=%s, n_library=%d, n_background=%d, "
                "domain_auc=%s)" % (self.mode, self.n_library, self.n_background,
                                    "None" if self.domain_auc is None
                                    else round(self.domain_auc, 3)))

    # ---- weights -------------------------------------------------------
    def raw_weights(self, smiles, keys=None):
        """SMILES (and, for the legacy box, their InChIKeys) -> raw, unconditioned
        weights. `nan` where the molecule is unparseable -- never 0, so that
        "cannot be featurised" stays distinguishable from "outside the box"."""
        smiles = list(smiles)
        keys = list(keys) if keys is not None else [None] * len(smiles)
        if len(keys) != len(smiles):
            raise ValueError("keys and smiles must be the same length")

        if self.mode == MODE_LEGACY_BOX:
            out = np.full(len(smiles), np.nan, dtype=np.float64)
            for i, (smi, k) in enumerate(zip(smiles, keys)):
                if k is not None and k in self.keys:      # exact InChIKey match
                    out[i] = 1.0
                    continue
                mol = parse_molecule(smi)
                if mol is None:
                    continue                              # stays nan -> dropped
                out[i] = 1.0 if in_envelope(mol, self.envelope) else 0.0
            return out

        if self.pipeline is None:
            raise RuntimeError("this subspace has no fitted domain classifier")
        X, ok = self._featurise_many(smiles)
        out = np.full(len(smiles), np.nan, dtype=np.float64)
        if ok.any():
            # X already holds only the featurizable rows, in order, so it is indexed
            # by nothing and scattered back through `ok`.
            p = self.pipeline.predict_proba(X)[:, 1]
            p = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)    # bound a separated odds
            out[ok] = p / (1.0 - p)
        return out

    def weights(self, smiles, keys=None, clip=None):
        """Conditioned loss weights (mean 1 over the survivors) + the kept mask."""
        clip = self.weight_clip if clip is None else clip
        return normalise_weights(self.raw_weights(smiles, keys=keys), clip=clip)

    def _featurise_many(self, smiles):
        """-> (rows for the featurizable compounds only, mask over the input)."""
        rows, ok = [], np.zeros(len(smiles), dtype=bool)
        for i, smi in enumerate(smiles):
            mol = parse_molecule(smi)
            if mol is None:
                continue
            row = self._featurise(mol)
            if row is None:
                continue
            rows.append(row)
            ok[i] = True
        if not rows:
            return np.zeros((0, 1), dtype=np.float64), ok
        return np.vstack(rows), ok

    def summary(self):
        """The A1/A3 half of the run report -- everything known before training."""
        return {
            "subspace_mode": self.mode,
            "domain_auc": None if self.domain_auc is None else round(float(self.domain_auc), 4),
            "n_library_sampled": self.n_library,
            "n_background": self.n_background,
            "classifier": self.classifier,
            "features": self.features,
            "weight_clip": self.weight_clip,
            "envelope_from_sample": self.envelope_from_sample,
        }


def in_envelope(mol, envelope):
    for (lo, hi), v in zip(envelope.values(), descriptors(mol)):
        if v < lo or v > hi:
            return False
    return True


# ---- the background set --------------------------------------------------
def load_fold_compounds(folds_dir, n_folds=5, fold_csv_template="train_%d.csv",
                        drug_col="InChiKey", smiles_col="SMILE"):
    """The AI-Bind training chemistry: unique (InChIKey, SMILES) over all folds.

    This is the classifier's negative class. It is pooled across the five folds on
    purpose: the density ratio describes "AI-Bind's chemistry versus the user's",
    which is a property of the pretraining corpus, not of any one split. It uses no
    labels and no target information, so the folds' independence as *training sets*
    (invariant 3) is untouched -- what stays per-fold is every fitting decision.
    """
    folds_dir = str(folds_dir).rstrip("/\\")
    seen = {}
    for r in range(n_folds):
        path = "%s/%s" % (folds_dir, fold_csv_template % r)
        df = pd.read_csv(path, usecols=[drug_col, smiles_col])
        for k, smi in zip(df[drug_col].values, df[smiles_col].values):
            if isinstance(smi, str) and k not in seen:
                seen[k] = smi
    return list(seen.keys()), list(seen.values())


# ---- A1: build the subspace ---------------------------------------------
def build_subspace(user_smiles, background_smiles=None, mode=DEFAULT_MODE,
                   classifier="logistic", features="descriptors",
                   max_library_sample=MAX_LIBRARY_SAMPLE,
                   max_background_sample=None, weight_clip=WEIGHT_CLIP,
                   n_bits=ECFP_BITS, radius=ECFP_RADIUS, cv_folds=5, seed=0,
                   inchikeys=None, verbose=False):
    """User library (+ the AI-Bind compounds) -> a fitted `ChemicalSubspace`.

    mode="density_ratio" (default)
        Fits `classifier` to separate `user_smiles` from `background_smiles` and
        reports the cross-validated AUC of that separation, which is A3's first
        diagnostic. Requires `background_smiles` -- there is no density ratio with
        only one density. Costs O(max_library_sample), not O(len(user_smiles)).

    mode="legacy_box"
        The pre-A1 behaviour: InChIKey set plus min/max envelope, hard 0/1 weights.
        Sampling is *disabled* in this mode, because the box is defined by the
        library's extremes -- sampling it would change its meaning, which is exactly
        the pathology the mode exists to demonstrate.

    classifier : "logistic" | "gbm"
    features   : "descriptors" | "ecfp" | "both"
        Descriptors are the default rather than fingerprints because a
        high-dimensional fingerprint separates 5.6k labeled compounds from a 0.5M
        library almost perfectly whatever the real overlap; that pushes the AUC to
        ~1 and the ESS to ~0 by capacity alone, and A3 would then read "near-disjoint
        chemistry" for every library it is ever shown. A dozen descriptors can only
        report a shift that is actually there on axes that actually matter.

    inchikeys : None -> True in legacy mode (required), False in density mode
        Hashing 0.5M InChIKeys buys nothing once nothing does exact matching. Pass
        True to get the library/AI-Bind overlap as well.
    """
    user_smiles = list(user_smiles)
    if not user_smiles:
        raise ValueError("empty compound library")
    mode = DEFAULT_MODE if mode is None else mode      # an R NULL means "the default"
    if mode not in (MODE_DENSITY_RATIO, MODE_LEGACY_BOX):
        raise ValueError("unknown subspace mode %r (expected %r or %r)"
                         % (mode, MODE_DENSITY_RATIO, MODE_LEGACY_BOX))
    if inchikeys is None:
        inchikeys = (mode == MODE_LEGACY_BOX)
    if mode == MODE_LEGACY_BOX:
        inchikeys = True                    # exact matching needs them
        max_library_sample = None           # the box must see the extremes

    idx = _sample_indices(len(user_smiles), max_library_sample, seed)
    keys, key2smi = set(), {}
    box_rows, dom_rows = [], []
    featurise = _make_featuriser(features, n_bits, radius)
    step = max(1, len(idx) // 10)

    for n_done, i in enumerate(idx, start=1):
        mol = parse_molecule(user_smiles[i])
        if mol is None:
            continue
        if inchikeys:
            k = Chem.MolToInchiKey(mol)
            keys.add(k)
            key2smi.setdefault(k, user_smiles[i])
        box_rows.append(descriptors(mol))
        if mode == MODE_DENSITY_RATIO:
            row = featurise(mol)
            if row is not None:
                dom_rows.append(row)
        if verbose and n_done % step == 0:
            print("  subspace: %d/%d library compounds" % (n_done, len(idx)), flush=True)

    if not box_rows:
        raise ValueError("no parseable SMILES in the user library")
    arr = np.asarray(box_rows)
    envelope = {p: (float(arr[:, i].min()), float(arr[:, i].max()))
                for i, p in enumerate(PROPS)}

    if mode == MODE_LEGACY_BOX:
        return ChemicalSubspace(MODE_LEGACY_BOX, keys, envelope, key2smi,
                                n_library=len(box_rows), weight_clip=weight_clip,
                                classifier=None, features=None)

    if background_smiles is None:
        raise ValueError(
            "mode='density_ratio' needs `background_smiles` (the AI-Bind training "
            "compounds; see load_fold_compounds()). Pass mode='legacy_box' for the "
            "pre-A1 min/max envelope, which needs only the library.")

    background_smiles = list(background_smiles)
    bg_idx = _sample_indices(len(background_smiles), max_background_sample, seed + 1)
    bg_rows = []
    for i in bg_idx:
        mol = parse_molecule(background_smiles[i])
        if mol is None:
            continue
        row = featurise(mol)
        if row is not None:
            bg_rows.append(row)
    if not bg_rows:
        raise ValueError("no parseable SMILES among the background compounds")
    if not dom_rows:
        raise ValueError("no library compound produced a usable descriptor row")

    X = np.vstack([np.vstack(dom_rows), np.vstack(bg_rows)])
    y = np.concatenate([np.ones(len(dom_rows), dtype=np.int64),
                        np.zeros(len(bg_rows), dtype=np.int64)])
    pipeline, auc = _fit_domain_classifier(X, y, classifier=classifier,
                                           cv_folds=cv_folds, seed=seed,
                                           verbose=verbose)
    return ChemicalSubspace(MODE_DENSITY_RATIO, keys, envelope, key2smi,
                            pipeline=pipeline, featurise=featurise, domain_auc=auc,
                            n_library=len(dom_rows), n_background=len(bg_rows),
                            envelope_from_sample=(len(idx) < len(user_smiles)),
                            weight_clip=weight_clip, classifier=classifier,
                            features=features)


def _make_featuriser(features, n_bits, radius):
    """mol -> feature row, or None. Closes over one Morgan generator so the
    generator is built once rather than per molecule."""
    if features == "descriptors":
        return domain_descriptors
    gen = _ecfp_generator(n_bits, radius)
    if features == "ecfp":
        def _fp(mol):
            return gen.GetFingerprintAsNumPy(mol).astype(np.float64)
        return _fp
    if features == "both":
        def _both(mol):
            d = domain_descriptors(mol)
            if d is None:
                return None
            return np.concatenate([d, gen.GetFingerprintAsNumPy(mol).astype(np.float64)])
        return _both
    raise ValueError("unknown features %r (expected 'descriptors', 'ecfp' or 'both')"
                     % (features,))


def _fit_domain_classifier(X, y, classifier="logistic", cv_folds=5, seed=0,
                           verbose=False):
    """-> (fitted pipeline, out-of-fold AUC).

    scikit-learn is imported here rather than at module scope so the baseline
    (tier 0a, no fine-tune) path keeps exactly the dependencies it had.

    The AUC is out-of-fold, not in-sample: A3 uses it as a gate, and an in-sample
    AUC from a flexible classifier is optimistic by construction, which would bias
    the gate towards "adaptation will work" precisely when it will not.

    `class_weight="balanced"` is what makes the posterior odds a density ratio with
    the sampling prior divided out -- without it every weight carries a constant
    factor of the library/background count ratio, and the ratio of two arbitrary
    sample sizes has no business in a loss.

    n_jobs is left at the default: this runs inside reticulate under R, where
    joblib's process pool is a reliable source of hangs, and five logistic fits on
    <=25k rows do not need it.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if classifier == "logistic":
        est = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                 solver="lbfgs", random_state=seed)
        pipeline = make_pipeline(StandardScaler(), est)
    elif classifier == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        pipeline = make_pipeline(
            HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                           random_state=seed))
    else:
        raise ValueError("unknown classifier %r (expected 'logistic' or 'gbm')"
                         % (classifier,))

    auc = None
    n_min = int(min(np.bincount(y)))
    if n_min >= cv_folds and cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        try:
            oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, oof))
        except Exception as exc:            # a degenerate fold must not kill the run
            warnings.warn("domain-classifier cross-validation failed (%s); "
                          "AUC unavailable" % exc, RuntimeWarning)
    pipeline.fit(X, y)
    if verbose:
        print("  domain classifier: %s on %d features, AUC=%s"
              % (classifier, X.shape[1], "n/a" if auc is None else round(auc, 3)),
              flush=True)
    return pipeline, auc


# ---- A1: the fold pairs, weighted rather than filtered -------------------
def extract_fold_pairs(fold_csv, subspace, envelope=None,
                       drug_col="InChiKey", target_col="target_aa_code",
                       label_col="Y", smiles_col="SMILE", weight_clip=None,
                       return_report=False):
    """One fold split -> (smiles, seqs, labels, weights[, report]).

    Every labeled row survives except those whose compound cannot be featurised;
    what varies is the weight. That is the A1 change: the fold is weighted, not
    filtered, so a library's influence is graded rather than a cut.

    Rows dropped, and only these:
      - weight 0 or nan. In density-ratio mode that means the SMILES is
        unparseable or descriptor-degenerate (invariant 4). In legacy-box mode it
        additionally means "outside the envelope", which reproduces the old filter
        exactly -- and because the survivors then all carry weight 1, the mean-1
        rescale leaves them at 1 and the old *unweighted* training set comes back
        bit for bit. That is the reduction, executable.

    Backward compatibility: the pre-A1 positional call
    `extract_fold_pairs(csv, user_keys, envelope)` is still accepted and is routed
    to legacy-box behaviour, loudly, so an old caller cannot silently get the new
    weights or (worse) a sample-derived box.
    """
    if envelope is not None:
        warnings.warn(
            "extract_fold_pairs(fold_csv, user_keys, envelope) is the pre-A1 call; "
            "running in legacy_box mode. Pass a ChemicalSubspace from "
            "build_subspace() for importance weighting.", DeprecationWarning,
            stacklevel=2)
        subspace = ChemicalSubspace(MODE_LEGACY_BOX, set(subspace), envelope, {})

    df = pd.read_csv(fold_csv)
    if smiles_col not in df.columns:
        raise ValueError(
            "fold split %s has no '%s' column; build it with "
            "build_assets.bake_folds()" % (fold_csv, smiles_col))

    uniq = df[[drug_col, smiles_col]].drop_duplicates(subset=[drug_col])
    u_keys = [k for k, s in zip(uniq[drug_col].values, uniq[smiles_col].values)
              if isinstance(s, str)]
    u_smiles = [s for s in uniq[smiles_col].values if isinstance(s, str)]

    raw = subspace.raw_weights(u_smiles, keys=u_keys)
    key2smi = dict(zip(u_keys, u_smiles))
    key2raw = dict(zip(u_keys, raw))

    sub = df[df[drug_col].isin(key2raw)]
    keys_by_row = sub[drug_col].values
    row_raw = np.fromiter((key2raw[k] for k in keys_by_row), dtype=np.float64,
                          count=len(keys_by_row))

    # Conditioning happens per fold, on the fold's own weights, so the five folds
    # remain independent of one another (invariant 3).
    clip = subspace.weight_clip if weight_clip is None else weight_clip
    w, kept = normalise_weights(row_raw, clip=clip)

    smiles = [key2smi[k] for k, ok in zip(keys_by_row, kept) if ok]
    seqs = [s for s, ok in zip(sub[target_col].values, kept) if ok]
    labels = [float(y) for y, ok in zip(sub[label_col].values, kept) if ok]
    weights = w[kept]

    if not return_report:
        return smiles, seqs, labels, weights
    report = {
        "rows_in": int(len(df)),
        "rows_kept": int(kept.sum()),
        "rows_dropped_zero_weight": int((~kept).sum()),
        "unique_compounds": int(len(u_smiles)),
        "unique_unweightable": int(np.count_nonzero(~np.isfinite(raw))),
        "ess": effective_sample_size(weights),
    }
    return smiles, seqs, labels, weights, report
