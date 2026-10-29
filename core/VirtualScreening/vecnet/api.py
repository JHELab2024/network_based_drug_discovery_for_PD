"""
The AI-component entry point.

Four user-facing choices, nothing else:
  - `targets`  : one protein sequence, or many
  - `library`  : the user's small-molecule library (SMILES)
  - `device`   : "cuda" / "cpu" / None (auto-detect, CPU fallback)
  - `finetune` : False -> pretrained cold-start model; True -> adapt to the library

Everything is a plain function with explicit arguments returning plain types
(numpy arrays, lists, dicts), so reticulate maps it to R without adapters:
a returned dict becomes a named list and the (n_drugs x n_targets) score array
becomes an R matrix.

    from vecnet import screen
    out = screen(targets=seq, library=smiles, device="cuda", finetune=False,
                 assets_root="/.../VirtualScreeningArm")
    out["scores"]        # (n_drugs, n_targets) P(bind)

Fit-once/predict-many is the class form (`VecNetScreener`): fine-tuning happens
once at fit time, never per target and never per call.

Tiers
-----
The same two inputs, three ways of using them:

  0a  no fine-tune -- the faithful port of the pretrained cold-start model
  0b  library-weighted adaptation under covariate shift (A1-A4)
  1   0b plus measured ChEMBL actives for the target -- **not implemented here**

`tier` is the forward-facing name for the choice `finetune` used to express, and
the two stay consistent: setting one sets the other. Asking for tier 1 is not an
error; it records `tier_requested="1"`, falls back to 0b and says why in
`tier_note`, which is the plan's rule that a user gets an honest label rather than
a failure.

The run report
--------------
`report` carries, per run and per target: `tier_requested`, `tier_run`,
`n_actives_found`, `mean_library_score`, `binding_n`, plus the fit diagnostics.
Fields that this build cannot fill -- `n_actives_found` and `binding_n` come from
the tier-1 ChEMBL fetch -- are present and None rather than absent, so a caller can
bind runs from different builds into one table without a schema mismatch.

Two diagnostics are free at screening time and always present: the mean
library score (below ~0.1 the target is outside the model's usable range) and the
per-compound disagreement between the five folds. The other two -- the
domain-classifier AUC and the effective sample size -- need the fold splits and the
library's descriptor distribution, so they are computed during `fit()`, which
`fit()` calls anyway and which can be run **on its own, before deciding to
fine-tune at all**. That is the point of A3: a gate, not a post-hoc explanation.
"""
import time
import warnings

import numpy as np

from .assets import Assets
from .cache import DrugEmbeddingCache
from .calibrate import calibrate_matrix, heavy_atom_counts
from .similarity import (similarity_scores, combine, calibrate_band,
                         DEFAULT_BAND, DEFAULT_BAND_GRID)

# Below this many known binders a calibration set cannot say anything: the
# grid is nine values wide and the enrichment of five compounds cannot
# separate them.
MIN_CALIBRATION_BINDERS = 10
# References cost linear time -- 1.82M compounds against 1,000 references is
# about 4 minutes, against 11,000 (ChEMBL's largest target) about 45. Cap by
# default, raise it when the wait is worth it.
DEFAULT_MAX_REFERENCES = 1000
from .device import describe, resolve_device
from .featurizer import Featurizer
from .finetune import (DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_VAL_FRACTION,
                       VecNetFineTuner)
from .predict import load_folds, score_targets
from .trainset import DEFAULT_MODE, MAX_LIBRARY_SAMPLE, WEIGHT_CLIP

TIERS = ("0a", "0b", "1")

# A3's existing usable-range gate. Below this the target sits outside the range the
# model was ever trained to discriminate in, and the ranking within the library is
# noise however clean it looks.
MEAN_SCORE_GATE = 0.1


# ---- input normalization -------------------------------------------------
def normalize_targets(targets, target_names=None):
    """str | list[str] | dict[name -> seq] -> (names, sequences).

    Accepts what R hands over: a character vector becomes a list, a named
    character vector becomes a dict.
    """
    if isinstance(targets, str):
        seqs, names = [targets], ["target_1"]
    elif isinstance(targets, dict):
        names = [str(k) for k in targets.keys()]
        seqs = [str(v) for v in targets.values()]
    else:
        seqs = [str(s) for s in targets]
        names = ["target_%d" % (i + 1) for i in range(len(seqs))]

    if target_names is not None:
        target_names = [target_names] if isinstance(target_names, str) else list(target_names)
        if len(target_names) != len(seqs):
            raise ValueError("target_names has %d entries but %d targets were given"
                             % (len(target_names), len(seqs)))
        names = [str(n) for n in target_names]

    if not seqs:
        raise ValueError("no target sequences given")
    for i, s in enumerate(seqs):
        if not s or not s.strip():
            raise ValueError("target %r has an empty sequence" % names[i])
    return names, [s.strip().upper() for s in seqs]


def normalize_library(library, library_ids=None):
    """list[str] | dict[id -> smiles] -> (ids, smiles)."""
    if isinstance(library, str):
        smiles, ids = [library], ["cpd_1"]
    elif isinstance(library, dict):
        ids = [str(k) for k in library.keys()]
        smiles = [str(v) for v in library.values()]
    else:
        smiles = [str(s) for s in library]
        ids = ["cpd_%d" % (i + 1) for i in range(len(smiles))]

    if library_ids is not None:
        library_ids = [library_ids] if isinstance(library_ids, str) else list(library_ids)
        if len(library_ids) != len(smiles):
            raise ValueError("library_ids has %d entries but the library has %d compounds"
                             % (len(library_ids), len(smiles)))
        ids = [str(i) for i in library_ids]

    if not smiles:
        raise ValueError("empty compound library")
    return ids, smiles


def read_fasta(path):
    """FASTA -> {header_id: sequence}. Convenience only; `targets` takes plain strings."""
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        out[name] = "".join(buf)
    if not out:
        raise ValueError("no records parsed from %s" % path)
    return out


def resolve_tier(tier, finetune):
    """(tier, finetune) -> (tier_requested, finetune, note).

    An explicit tier wins over `finetune`, because it is the more specific
    statement; passing neither keeps the old behaviour exactly.
    """
    if tier is None:
        return ("0b" if finetune else "0a"), bool(finetune), None
    tier = str(tier).strip().lower()
    if tier not in TIERS:
        raise ValueError("unknown tier %r (expected one of %s)" % (tier, ", ".join(TIERS)))
    if tier == "0a":
        return "0a", False, None
    if tier == "0b":
        return "0b", True, None
    return ("1", True,
            "tier 1 (measured ChEMBL actives for the target) is not implemented in "
            "this build; ran tier 0b instead")


# ---- the screener --------------------------------------------------------
class VecNetScreener:
    """Fit-once / predict-many. `screen()` below wraps this for one-shot use."""

    def __init__(self, assets=None, assets_root=None, device=None, finetune=False,
                 cache_dir=None, verbose=False, tier=None):
        if assets is None:
            if assets_root is None:
                raise ValueError("pass assets_root (or a prebuilt Assets object)")
            assets = Assets.from_root(assets_root)
        self.tier_requested, finetune, self.tier_note = resolve_tier(tier, finetune)
        assets.require(finetune=finetune)

        self.assets = assets
        self.finetune = bool(finetune)
        self.device = resolve_device(device)
        self.verbose = verbose
        self._say("device: %s" % describe(self.device))
        if self.tier_note:
            self._say(self.tier_note)

        t0 = time.time()
        self.feat = Featurizer(assets.mol2vec_path, assets.protvec_csv)
        self._say("featurizer loaded in %.1fs" % (time.time() - t0))

        # cache_dir: None -> the default location under assets; False -> in-memory
        # only; a path -> there. `None or assets.cache_dir` would make "off"
        # unexpressible, so False is honoured explicitly.
        resolved_cache = assets.cache_dir if cache_dir is None else cache_dir
        self.cache = DrugEmbeddingCache(self.feat, resolved_cache)
        self.bundle_npz = assets.bundle_npz()      # baseline until fit() replaces it
        self.tuner = None
        self.models = None
        self.fit_report_ = []
        self.diagnostics_ = {}
        self.fitted_ = False

    def _say(self, msg):
        if self.verbose:
            print("[vecnet] %s" % msg, flush=True)

    def _ensure_tuner(self):
        """The fine-tuner owns the subspace preparation, so it is built for a fit()
        that never trains. Its fold models load lazily, so this costs a featurizer
        reference and nothing else."""
        self.assets.require(finetune=True)
        if self.tuner is None:
            self.tuner = VecNetFineTuner(self.assets.bundle_npz(), self.feat,
                                         self.assets.folds_dir, self.device)
        return self.tuner

    # ---- A3: the gate, runnable without training -------------------------

    def fit(self, library, tuned_dir=None, epochs=None, max_epochs=None,
            lr=1e-4, weight_decay=1e-2, batch_size=256, seed=0,
            early_stopping=True, val_fraction=DEFAULT_VAL_FRACTION,
            patience=DEFAULT_PATIENCE, subspace_mode=None,
            classifier="logistic", features="descriptors",
            max_library_sample=MAX_LIBRARY_SAMPLE, weight_clip=WEIGHT_CLIP,
            diagnostics=True):
        """Adapt to `library` under covariate shift. No-op when finetune=False.

        `epochs` is the pre-A4 argument and still works, as a *maximum*: early
        stopping then selects within it, and `epochs=0` keeps its exact old meaning
        of touching nothing.
        """
        if not self.finetune:
            return self
        _, smiles = normalize_library(library)
        t0 = time.time()
        # None means "the default" throughout, so an R NULL never reaches arithmetic.
        max_epochs = DEFAULT_MAX_EPOCHS if max_epochs is None else max_epochs
        subspace_mode = DEFAULT_MODE if subspace_mode is None else subspace_mode
        n_ep = max_epochs if epochs is None else epochs
        self._say("fine-tuning on %d compounds (mode=%s, max_epochs=%s, lr=%g)"
                  % (len(smiles), subspace_mode, n_ep, lr))
        self._ensure_tuner()
        self.tuner.fit(smiles, max_epochs=max_epochs, epochs=epochs, lr=lr,
                       weight_decay=weight_decay, batch_size=batch_size, seed=seed,
                       early_stopping=early_stopping, val_fraction=val_fraction,
                       patience=patience, mode=subspace_mode, classifier=classifier,
                       features=features, max_library_sample=max_library_sample,
                       weight_clip=weight_clip, verbose=self.verbose)
        self.fit_report_ = self.tuner.fit_report_
        self.diagnostics_ = self.tuner.diagnostics_
        self.fitted_ = n_ep > 0
        if tuned_dir:
            self.bundle_npz = self.tuner.save(tuned_dir)
            self._say("tuned bundle written to %s" % self.bundle_npz)
        self.models = self.tuner.models
        self._say("fine-tune done in %.1fs" % (time.time() - t0))
        return self

    def load(self, bundle_npz=None):
        """Load fold models for inference (baseline, or a tuned bundle)."""
        self.models = load_folds(bundle_npz or self.bundle_npz, self.device)
        return self

    def _score_extra(self, smiles, seqs, batch_size):
        """Score compounds that are not part of the library being ranked.

        Used by similarity_band="auto": the known binders need sequence-model scores
        on the same scale as the library to calibrate against it, but they must not
        appear in the nomination list. Scoring them here keeps them out of it.
        """
        raw_drug, valid = self.cache.encode(smiles, verbose=False)
        raw_targets = [self.feat.target_raw(s) for s in seqs]
        return score_targets(self.models, raw_drug, valid, raw_targets,
                             self.device, batch_size=batch_size, verbose=False)

    def screen(self, targets, library, target_names=None, library_ids=None,
               batch_size=4096, fold_spread=True, calibrate=False,
               reference_ligand=None, similarity_band=None,
               known_binders=None, calibration_fraction=0.01,
               max_references=DEFAULT_MAX_REFERENCES):
        """-> dict(ids, smiles, targets, scores (n_drugs, n_targets), valid, report).

        fold_spread : also return the per-compound across-fold standard deviation
            and its summary in the report (A3). Nearly free -- the five fold
            probabilities are computed either way.
        calibrate   : also return A2's `percentile`, `size_residual` and
            `residual_rank` columns. Off by default because it re-parses the library
            for heavy-atom counts; on, it is the ranking that should actually be
            used, and the only one comparable across targets.

        similarity_band : resolution of the ligand-similarity channel, or "auto"
            to choose it per target from `known_binders`. "auto" needs a handful of
            compounds already known to bind -- a list, or a dict keyed by target.
            They are scored alongside the library, the band that best enriches them
            is selected, and they are then removed from the returned ranking: they
            calibrate the parameter, they are never nominated. Because w = 1.0 (the
            sequence ranking) is in the grid, "auto" cannot pick a band that does
            worse than the sequence model alone on the calibration compounds.
        """
        names, seqs = normalize_targets(targets, target_names)
        ids, smiles = normalize_library(library, library_ids)
        if self.models is None:
            self.load()

        t0 = time.time()
        raw_drug, valid = self.cache.encode(smiles, verbose=self.verbose)
        t_feat = time.time() - t0
        self._say("featurization: %.1fs (%d/%d parseable)" % (t_feat, int(valid.sum()), len(smiles)))

        raw_targets = [self.feat.target_raw(s) for s in seqs]

        spread = (np.full((len(smiles), len(seqs)), np.nan, dtype=np.float64)
                  if fold_spread else None)
        t0 = time.time()
        scores = score_targets(self.models, raw_drug, valid, raw_targets,
                               self.device, batch_size=batch_size,
                               verbose=self.verbose, spread=spread)
        t_score = time.time() - t0
        self._say("scoring: %.1fs for %d targets" % (t_score, len(seqs)))

        # ---- optional ligand-similarity channel ------------------------------
        #
        # Four modes, and which one runs is decided by what the caller supplied:
        #
        #   nothing               -> sequence only; no similarity is computed
        #   known_binders         -> channel on, band chosen per target ("auto")
        #   similarity_band=<w>   -> channel on at a fixed w (0 = similarity only,
        #                            1 = the sequence ranking exactly)
        #   reference_ligand      -> use that compound as the reference instead of
        #                            the binders; the pre-bundle path, kept working
        #
        # The references default to the known binders themselves. A target's chemistry
        # is rarely one scaffold, and a library compound only has to resemble ONE
        # thing already known to bind, so several references beat a single ligand.
        similarity = combined = None
        bands = {}
        binders = {}
        if known_binders:
            binders = (dict(known_binders) if isinstance(known_binders, dict)
                       else {n: list(known_binders) for n in names})
        # A reference may be given per target, for all targets, or not at all. An
        # empty entry is treated as absent, not as an empty molecule -- the bundle
        # carries a crystal ligand for only seven targets and a blank for the rest.
        given = {}
        if reference_ligand is not None:
            given = (reference_ligand if isinstance(reference_ligand, dict)
                     else {n: reference_ligand for n in names})
        external = {n: given[n] for n in names if given.get(n)}
        refs = {}
        for n in names:
            if n in external:
                refs[n] = external[n]
            elif binders.get(n):
                refs[n] = binders[n]
        if given and not external and not binders:
            raise ValueError("reference_ligand was given but no usable reference "
                             "could be parsed for any target")

        if refs:
            auto = (similarity_band is None
                    or (isinstance(similarity_band, str)
                        and similarity_band.lower() == "auto"))
            if auto and not binders:
                # A lone reference ligand cannot be calibrated -- there is nothing to
                # measure enrichment on. Fall back to the shipped default rather than
                # refuse, but say so: on the paper's application that fixed band was
                # the wrong choice for six targets out of seven.
                auto = False
                similarity_band = DEFAULT_BAND
                warnings.warn("similarity_band defaulted to %g because no "
                              "known_binders were given to calibrate with; supply "
                              "binders to let the band be chosen per target"
                              % DEFAULT_BAND, stacklevel=2)
            if auto:
                cal_smiles = sorted({x for n in names for x in binders.get(n, [])})
                cal_index = {x: i for i, x in enumerate(cal_smiles)}
                cal_scores = self._score_extra(cal_smiles, seqs, batch_size)

            similarity = np.full((len(smiles), len(names)), np.nan, dtype=np.float64)
            combined = np.full_like(similarity, np.nan)
            for j, n in enumerate(names):
                t0 = time.time()
                mine = binders.get(n, [])
                if auto:
                    if len(mine) < MIN_CALIBRATION_BINDERS:
                        raise ValueError(
                            "target %s has %d known binder(s); calibrating a band "
                            "needs at least %d. Pass an explicit similarity_band, or "
                            "leave the channel off." % (n, len(mine),
                                                        MIN_CALIBRATION_BINDERS))
                    if n in external:
                        # The reference is external to the binders, so every binder
                        # can serve as a calibration positive.
                        cal_refs, cal_rows = refs[n], list(mine)
                    else:
                        # The references ARE binders, so they must be held out of the
                        # positives: a compound's similarity to itself is 1.0 and
                        # would make w = 0 win every time. Half and half.
                        half = len(mine) // 2
                        cal_refs, cal_rows = mine[:half], mine[half:]
                    rows = [cal_index[x] for x in cal_rows]
                    cal_sim = similarity_scores(cal_refs, cal_rows,
                                                max_references=max_references)
                    lib_sim = similarity_scores(cal_refs, smiles,
                                                max_references=max_references)
                    y = np.concatenate([np.zeros(len(smiles), bool),
                                        np.ones(len(rows), bool)])
                    band, table = calibrate_band(
                        np.concatenate([scores[:, j], cal_scores[rows, j]]),
                        np.concatenate([lib_sim, cal_sim]), y,
                        fraction=calibration_fraction)
                    bands[n] = {"band": band, "n_calibration": len(rows),
                                "n_references": len(cal_refs), "grid": table}
                    self._say("band for %s: %g (chosen from %d binders against %d "
                              "references)" % (n, band, len(rows), len(cal_refs)))
                else:
                    band = float(similarity_band)
                    bands[n] = {"band": band, "n_calibration": 0,
                                "n_references": None, "grid": None}
                # Screen with every reference available, not just the half used to
                # calibrate. Similarity is therefore stronger at screening time than
                # it was at calibration time, which biases the chosen band toward the
                # sequence model -- the safe direction.
                similarity[:, j] = similarity_scores(refs[n], smiles,
                                                     max_references=max_references)
                combined[:, j] = combine(scores[:, j], similarity[:, j], band=band)
                self._say("similarity channel for %s: %.1fs (band=%g)"
                          % (n, time.time() - t0, band))

        tier_run, tier_note = self._tier_run()
        result = {
            "ids": ids,
            "smiles": smiles,
            "targets": names,
            "scores": scores,
            "similarity": similarity,
            "combined": combined,
            "rank_on": "combined" if combined is not None else "scores",
            "similarity_band": bands or None,
            "valid": valid,
            "fold_spread": spread,
            "report": {
                "n_compounds": len(smiles),
                "n_parseable": int(valid.sum()),
                "n_unparseable": int((~valid).sum()),
                "n_targets": len(seqs),
                "device": describe(self.device),
                "finetuned": self.finetune and self.fitted_,
                "bundle": self.bundle_npz,
                "seconds_featurize": round(t_feat, 2),
                "seconds_score": round(t_score, 2),
                "fit_report": self.fit_report_,
                "tier_requested": self.tier_requested,
                "tier_run": tier_run,
                "tier_note": tier_note,
                "diagnostics": self.diagnostics_,
                "targets": [self._target_row(names[j], scores[:, j],
                                             None if spread is None else spread[:, j],
                                             tier_run, tier_note)
                            for j in range(len(seqs))],
            },
        }

        if calibrate:
            cal = calibrate_matrix(scores, heavy_atoms=heavy_atom_counts(smiles),
                                   target_names=names)
            result["percentile"] = cal["percentile"]
            result["size_residual"] = cal["size_residual"]
            result["residual_rank"] = cal["residual_rank"]
            result["heavy_atoms"] = cal["heavy_atoms"]
            for row, rep in zip(result["report"]["targets"], cal["report"]):
                row["spearman_size"] = rep.get("spearman_size")
                row["size_r2"] = rep.get("r2_size")
        return result

    # ---- report helpers --------------------------------------------------
    def _tier_run(self):
        """What actually ran, which is not always what was asked for."""
        note = self.tier_note
        if not self.finetune:
            return "0a", note
        if not self.fitted_:
            extra = ("a fine-tune was requested but none has run (fit() not called, "
                     "or max_epochs=0), so this is the pretrained model")
            return "0a", "; ".join([n for n in (note, extra) if n])
        return "0b", note

    def _target_row(self, name, col, spread_col, tier_run, tier_note):
        """The per-target half of the reporting contract.

        `n_actives_found` and `binding_n` come from the tier-1 ChEMBL fetch, which
        this build does not implement. They are None rather than absent so that runs
        from different builds bind into one table.
        """
        finite = np.isfinite(col)
        mean_score = float(col[finite].mean()) if finite.any() else float("nan")
        row = {
            "target": name,
            "tier_requested": self.tier_requested,
            "tier_run": tier_run,
            "tier_note": tier_note,
            "n_actives_found": None,
            "binding_n": None,
            "n_scored": int(finite.sum()),
            "mean_library_score": None if not finite.any() else round(mean_score, 6),
            "usable_range": bool(finite.any() and mean_score >= MEAN_SCORE_GATE),
            "mean_score_gate": MEAN_SCORE_GATE,
        }
        if spread_col is not None:
            ok = np.isfinite(spread_col)
            row.update({
                "fold_disagreement_mean": round(float(spread_col[ok].mean()), 6) if ok.any() else None,
                "fold_disagreement_median": round(float(np.median(spread_col[ok])), 6) if ok.any() else None,
                "fold_disagreement_p95": round(float(np.quantile(spread_col[ok], 0.95)), 6) if ok.any() else None,
            })
        return row


# ---- one-shot function ---------------------------------------------------
def screen(targets, library, assets_root=None, assets=None, device=None,
           finetune=False, target_names=None, library_ids=None, tuned_dir=None,
           cache_dir=None, batch_size=4096, verbose=False,
           finetune_epochs=None, finetune_max_epochs=None,
           finetune_lr=1e-4, finetune_seed=0, tier=None, subspace_mode=None,
           fold_spread=True, calibrate=False,
           reference_ligand=None, similarity_band=None,
           known_binders=None, calibration_fraction=0.01,
           max_references=DEFAULT_MAX_REFERENCES):
    """Screen a library against one or many targets.

    finetune=False -> pretrained cold-start model (needs bundle + encoders only)
    finetune=True  -> adapt drug_proj+head to the library first (needs fold splits);
                      the fit happens once and is reused across every target.

    cache_dir: None -> default under assets; False -> in-memory only; path -> there.

    Two defaults changed with A1/A4:
      - `finetune_epochs` is None, i.e. no fixed epoch count. Early stopping selects
        within `finetune_max_epochs` on held-out weighted validation loss, and the
        pretrained model (epoch 0) is inside that search space. Pass an integer to
        pin a maximum, as the old `finetune_epochs=5` did.
      - `subspace_mode` is "density_ratio": the library weights the training pairs
        rather than filtering them. "legacy_box" restores the pre-A1 min/max
        envelope.
    """
    subspace_mode = DEFAULT_MODE if subspace_mode is None else subspace_mode
    finetune_max_epochs = (DEFAULT_MAX_EPOCHS if finetune_max_epochs is None
                           else finetune_max_epochs)
    s = VecNetScreener(assets=assets, assets_root=assets_root, device=device,
                       finetune=finetune, cache_dir=cache_dir, verbose=verbose,
                       tier=tier)
    if s.finetune:
        s.fit(library, tuned_dir=tuned_dir, epochs=finetune_epochs,
              max_epochs=finetune_max_epochs, lr=finetune_lr, seed=finetune_seed,
              subspace_mode=subspace_mode)
    return s.screen(targets, library, target_names=target_names,
                    library_ids=library_ids, batch_size=batch_size,
                    fold_spread=fold_spread, calibrate=calibrate,
                    reference_ligand=reference_ligand,
                    similarity_band=similarity_band,
                    known_binders=known_binders,
                    calibration_fraction=calibration_fraction,
                    max_references=max_references)


def top_hits(result, target=None, n=20, rank_by="score"):
    """Convenience: the n best-scoring compounds for one target, as a plain dict.

    rank_by="score"    : raw P(bind), the pre-A2 behaviour and the default.
    rank_by="residual" : A2's size residual of the library percentile, available
        when the screen was run with calibrate=True. This is the ranking to use for
        anything compared across targets, because the raw score is not comparable
        across them and is correlated with molecular size in a direction the target
        sets.
    """
    names = result["targets"]
    j = 0 if target is None else (names.index(target) if isinstance(target, str) else int(target))
    if rank_by == "score":
        col = result["scores"][:, j]
    elif rank_by == "residual":
        if "size_residual" not in result:
            raise ValueError("rank_by='residual' needs a screen run with calibrate=True")
        col = result["size_residual"][:, j]
    else:
        raise ValueError("rank_by must be 'score' or 'residual'")
    order = np.argsort(np.where(np.isnan(col), -np.inf, col))[::-1][:n]
    out = {"target": names[j],
           "ids": [result["ids"][i] for i in order],
           "smiles": [result["smiles"][i] for i in order],
           "score": result["scores"][:, j][order]}
    for key in ("percentile", "size_residual"):
        if key in result:
            out[key] = result[key][:, j][order]
    return out
