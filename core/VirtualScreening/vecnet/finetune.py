"""
VecNetFineTuner -- adapt the pretrained VecNet to the distribution a user's library
describes, under covariate shift.

    ft = VecNetFineTuner(bundle_npz, featurizer, folds_dir, device)
    ft.fit(library_smiles)          # weighted adaptation with early stopping
    ft.save(out_dir)                # pretrained-format bundle, drop-in for load_folds()

Per fold r = 0..4 (folds stay five and stay independent):
  1. weight that fold's labeled pairs by the density ratio the library implies
     (A1, see trainset.py) -- the fold is weighted, not filtered;
  2. featurize each pair once (cached across epochs and folds);
  3. hold out a stratified validation split of the same pairs, with the same
     weights (A4);
  4. train `drug_proj` + `head` with `target_proj` and the four norm buffers frozen
     (weighted BCEWithLogitsLoss, AdamW, low LR), stopping on weighted validation
     loss rather than after a fixed five epochs.

What changed, and why it matters
--------------------------------
A1: the loss is now `mean(w * BCE)` with `w` the importance weight of the pair's
compound. Weights arrive already conditioned to mean 1 over the fold, so the loss
sits on the same scale as the unweighted baseline and `lr` keeps its meaning.
`fit_report_` carries the effective sample size, because a weighted fit on 60k
pairs with ESS 800 is an 800-pair fit and should be read as one.

A4: five epochs was a magic number. Selection now happens on held-out weighted
validation loss, and **epoch 0 is in the search space** -- the pre-training
snapshot is taken before the optimizer's first step, so if no epoch beats it the
fold's weights are restored to exactly the tensors that were loaded from the
pretrained bundle. That is what turns "the adaptation is bounded below by the
pretrained model" from a hope into a property of the code.

Invariants this file must not break
-----------------------------------
  1. **zero-epoch identity**: `fit(max_epochs=0)` returns before touching any model,
     so `save()` -> `predict` is bit-identical to the baseline. Diagnostics are a
     `prepare()` precisely so that this path can stay empty.
  2. **target_proj and the four constants never move.** `set_finetune_mode()`
     freezes target_proj, the buffers are never in `trainable_parameters()`, and the
     early-stopping snapshot/restore covers the *whole* state_dict, so a restore can
     only ever put them back exactly as they were.
  3. **Five folds, independent.** Each fold reads its own split, conditions its own
     weights, makes its own validation split (seed + r) and gets its own optimizer.
     The only thing shared is the label-free weight function.
  4. **The all-zero drug-vector guard.** `_drug()` still rejects blank and 0-atom
     molecules, and any pair whose drug it rejects is removed in `prepare()` before
     it can become a training example.
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

from .assets import BUNDLE_FILENAME
from .device import resolve_device
from .model import VecNet
from .trainset import (DEFAULT_MODE, MAX_LIBRARY_SAMPLE, MODE_LEGACY_BOX,
                       WEIGHT_CLIP, build_subspace, effective_sample_size,
                       extract_fold_pairs, load_fold_compounds)

N_FOLDS = 5

# The ceiling on the search, not a target. Early stopping picks the epoch; this only
# bounds how long it may look. It replaces the fixed 5 of the pre-A4 version, which
# was the number of epochs actually run.
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 2
DEFAULT_VAL_FRACTION = 0.2
MIN_VAL_PAIRS = 8          # below this a validation loss is noise, not a criterion


def _rescale_mean1(w):
    """Restore mean 1 after rows have been dropped.

    Not the full median/clip/rescale of `normalise_weights`: the clip was applied to
    the raw odds and stays valid: only the mean-1 property, which keeps the loss on
    the baseline's scale, needs re-establishing.
    """
    w = np.asarray(w, dtype=np.float64)
    m = float(w.mean()) if w.size else 0.0
    return w / m if m > 0 else w


def _library_signature(smiles, params):
    """Content hash of (library, preparation parameters).

    Deliberately not `cache.library_key`: that key is versioned by the *drug
    embedding* cache's semantics, and bumping the embedding cache version has no
    business invalidating a prepared training set.
    """
    h = hashlib.md5()
    h.update(repr(params).encode("utf-8"))
    for s in smiles:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class VecNetFineTuner:
    def __init__(self, bundle_npz, featurizer, folds_dir, device=None,
                 fold_csv_template="train_%d.csv"):
        self.device = resolve_device(device)
        self.feat = featurizer
        self.bundle_npz = str(bundle_npz)
        self.folds_dir = str(folds_dir).rstrip("/\\")
        self.fold_csv_template = fold_csv_template
        self._models = None     # built on first access: `prepare()` needs none
        self._raw_drug = {}     # smiles -> 300d raw
        self._raw_target = {}   # sequence -> 100d raw
        self._prepared = None
        self.subspace_ = None
        self.fit_report_ = []
        self.diagnostics_ = {}

    @property
    def models(self):
        """The five fold models, loaded on demand.

        Lazy because the A3 gate is meant to be runnable before -- and instead of --
        a fine-tune, and there is no reason a diagnostic should pay for five copies
        of the network on the GPU.
        """
        if self._models is None:
            z = np.load(self.bundle_npz, allow_pickle=True)
            self._models = [VecNet().to(self.device).load_bundle(z, r)
                            for r in range(N_FOLDS)]
        return self._models

    # ---- cached featurization (training set) -----------------------------
    def _drug(self, smiles):
        """Raw drug vector, or None. The all-zero guard mirrors DrugEmbeddingCache:
        an empty (0-atom) molecule must never become a training example."""
        if smiles not in self._raw_drug:
            v = self.feat.drug_raw(smiles) if isinstance(smiles, str) and smiles.strip() else None
            self._raw_drug[smiles] = None if (v is None or not v.any()) else v
        return self._raw_drug[smiles]

    def _target(self, seq):
        if seq not in self._raw_target:
            self._raw_target[seq] = self.feat.target_raw(seq)
        return self._raw_target[seq]

    def prepare(self, library_smiles, mode=DEFAULT_MODE, classifier="logistic",
                features="descriptors", max_library_sample=MAX_LIBRARY_SAMPLE,
                weight_clip=WEIGHT_CLIP, seed=0, verbose=False, force=False):
        """Subspace -> weighted, featurizable pairs per fold -> diagnostics.

        Cached on a content hash of the library and these parameters, so calling
        `prepare()` and then `fit()` costs one preparation, not two. Touches no
        model weights, which is what lets `fit(max_epochs=0)` stay a no-op while the
        report still carries diagnostics.
        """
        library_smiles = list(library_smiles)
        mode = DEFAULT_MODE if mode is None else mode
        params = (mode, classifier, features, max_library_sample, weight_clip, seed,
                  self.folds_dir, self.fold_csv_template)
        sig = _library_signature(library_smiles, params)
        if self._prepared is not None and self._prepared["signature"] == sig and not force:
            return self.diagnostics_

        background = None
        if mode != MODE_LEGACY_BOX:
            _, background = load_fold_compounds(
                self.folds_dir, n_folds=N_FOLDS,
                fold_csv_template=self.fold_csv_template)
            if verbose:
                print("  background: %d unique AI-Bind training compounds"
                      % len(background), flush=True)

        subspace = build_subspace(library_smiles, background_smiles=background,
                                  mode=mode, classifier=classifier,
                                  features=features,
                                  max_library_sample=max_library_sample,
                                  weight_clip=weight_clip, seed=seed,
                                  verbose=verbose)
        self.subspace_ = subspace
        if verbose:
            print("  %r" % (subspace,), flush=True)

        folds, fold_reports = [], []
        for r in range(N_FOLDS):
            fold_csv = "%s/%s" % (self.folds_dir, self.fold_csv_template % r)
            smiles, seqs, labels, weights, rep = extract_fold_pairs(
                fold_csv, subspace, return_report=True)

            # Invariant 4, second gate: a pair whose drug cannot be featurized is
            # removed here, before it can reach a tensor. The first gate (blank /
            # 0-atom SMILES) already ran in trainset.parse_molecule.
            keep = [i for i, smi in enumerate(smiles) if self._drug(smi) is not None]
            n_unfeaturizable = len(smiles) - len(keep)
            smiles = [smiles[i] for i in keep]
            seqs = [seqs[i] for i in keep]
            labels = np.asarray([labels[i] for i in keep], dtype=np.float64)
            weights = _rescale_mean1(weights[keep])

            ess = effective_sample_size(weights)
            fold = {"fold": r, "smiles": smiles, "seqs": seqs, "labels": labels,
                    "weights": weights}
            folds.append(fold)
            fold_reports.append({
                "fold": r,
                "pairs": int(len(labels)),
                "positives": int(labels.sum()) if len(labels) else 0,
                "rows_in": rep["rows_in"],
                "dropped_zero_weight": rep["rows_dropped_zero_weight"],
                "dropped_unfeaturizable": int(n_unfeaturizable),
                "ess": round(ess, 1),
                "ess_fraction": round(ess / len(labels), 4) if len(labels) else 0.0,
                "weight_p50": round(float(np.median(weights)), 4) if len(weights) else 0.0,
                "weight_p95": round(float(np.quantile(weights, 0.95)), 4) if len(weights) else 0.0,
                "weight_max": round(float(weights.max()), 4) if len(weights) else 0.0,
            })
            if verbose:
                print("  fold %d: %d pairs (%d positive), ESS %.0f"
                      % (r, len(labels), int(labels.sum()) if len(labels) else 0, ess),
                      flush=True)

        all_w = np.concatenate([f["weights"] for f in folds]) if folds else np.zeros(0)
        n_pairs = int(sum(len(f["labels"]) for f in folds))
        pooled_ess = effective_sample_size(all_w)

        diagnostics = dict(subspace.summary())
        diagnostics.update({
            "n_library": len(library_smiles),
            "n_pairs": n_pairs,
            "n_positives": int(sum(int(f["labels"].sum()) for f in folds)),
            "ess": round(pooled_ess, 1),
            "ess_fraction": round(pooled_ess / n_pairs, 4) if n_pairs else 0.0,
            "folds": fold_reports,
        })
        self._prepared = {"signature": sig, "folds": folds}
        self.diagnostics_ = diagnostics
        return diagnostics

    # ---- fit -------------------------------------------------------------
    def fit(self, library_smiles, max_epochs=DEFAULT_MAX_EPOCHS, epochs=None,
            lr=1e-4, weight_decay=1e-2, batch_size=256, seed=0,
            early_stopping=True, val_fraction=DEFAULT_VAL_FRACTION,
            patience=DEFAULT_PATIENCE, min_delta=0.0, mode=DEFAULT_MODE,
            classifier="logistic", features="descriptors",
            max_library_sample=MAX_LIBRARY_SAMPLE, weight_clip=WEIGHT_CLIP,
            verbose=False):
        """Weighted adaptation with early stopping.

        `epochs` is the pre-A4 name and is honoured as a *maximum* -- so an old
        caller passing `epochs=5` gets at most five epochs and keeps the shorter one
        if it validates better, and `epochs=0` keeps its exact old meaning.
        """
        if epochs is not None:
            max_epochs = int(epochs)
        if max_epochs is None:
            max_epochs = DEFAULT_MAX_EPOCHS
        if max_epochs <= 0:
            # Zero-epoch identity (invariant 1). Return before touching anything --
            # not even `self.models`, which would load the bundle. Diagnostics for a
            # zero-epoch run come from `prepare()`, which is a separate call for
            # exactly this reason.
            return self

        self.prepare(library_smiles, mode=mode, classifier=classifier,
                     features=features, max_library_sample=max_library_sample,
                     weight_clip=weight_clip, seed=seed, verbose=verbose)
        torch.manual_seed(seed)
        self.fit_report_ = []
        bce = nn.BCEWithLogitsLoss(reduction="none")

        for r in range(N_FOLDS):
            fold = self._prepared["folds"][r]
            n = len(fold["labels"])
            record = {"fold": r, "pairs": n,
                      "positives": int(fold["labels"].sum()) if n else 0,
                      "ess": round(effective_sample_size(fold["weights"]), 1)}
            if n == 0:
                record.update({"epochs_run": 0, "best_epoch": 0,
                               "selected_pretrained": True,
                               "note": "no pairs in this fold's subspace"})
                self.fit_report_.append(record)
                if verbose:
                    print("  fold %d: no pairs, left at the pretrained weights" % r,
                          flush=True)
                continue

            Xt = torch.tensor(np.asarray([self._target(s) for s in fold["seqs"]]),
                              dtype=torch.float64, device=self.device)
            Xd = torch.tensor(np.asarray([self._drug(s) for s in fold["smiles"]]),
                              dtype=torch.float64, device=self.device)
            Y = torch.tensor(fold["labels"], dtype=torch.float32,
                             device=self.device).view(-1, 1)
            W = torch.tensor(fold["weights"], dtype=torch.float32,
                             device=self.device).view(-1, 1)

            tr_idx, va_idx = _stratified_split(
                fold["labels"], val_fraction if early_stopping else 0.0, seed + r)
            use_val = early_stopping and len(va_idx) >= MIN_VAL_PAIRS and len(tr_idx) > 0
            if early_stopping and not use_val:
                record["note"] = ("fold too small for a usable validation split; "
                                  "ran the full max_epochs without selection")
                tr_idx = np.arange(n)
                va_idx = np.zeros(0, dtype=np.int64)
            tr = torch.as_tensor(tr_idx, dtype=torch.long, device=self.device)
            va = torch.as_tensor(va_idx, dtype=torch.long, device=self.device)

            model = self.models[r]
            model.set_finetune_mode()
            opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr,
                                    weight_decay=weight_decay)

            # Epoch 0 is a candidate: the snapshot is taken before the optimizer's
            # first step, so restoring it returns the exact tensors loaded from the
            # pretrained bundle. This is what puts the pretrained model inside the
            # search space (A4) and hence bounds the adaptation below by it.
            best_state = _snapshot(model)
            best_loss = (_weighted_loss(model, bce, Xt, Xd, Y, W, va)
                         if use_val else float("inf"))
            best_epoch, bad, history = 0, 0, [best_loss]
            n_tr = int(tr.numel())

            for epoch in range(1, max_epochs + 1):
                model.train()
                perm = tr[torch.randperm(n_tr, device=self.device)]
                for s in range(0, n_tr, batch_size):
                    idx = perm[s:s + batch_size]
                    opt.zero_grad()
                    # A1: weight each pair's loss by its importance weight. The mean
                    # (rather than a per-batch self-normalisation) is deliberate --
                    # the weights already have mean 1 over the fold, so the epoch's
                    # gradient is the weighted-risk gradient and a batch of atypical
                    # chemistry contributes proportionally less rather than being
                    # rescaled back up to full influence.
                    loss = (W[idx] * bce(model(Xt[idx], Xd[idx]), Y[idx])).mean()
                    loss.backward()
                    opt.step()

                if not use_val:
                    continue
                vl = _weighted_loss(model, bce, Xt, Xd, Y, W, va)
                history.append(vl)
                if vl < best_loss - min_delta:
                    best_loss, best_epoch, bad = vl, epoch, 0
                    best_state = _snapshot(model)
                else:
                    bad += 1
                    if bad >= patience:
                        break

            epochs_run = epoch if max_epochs else 0
            if use_val:
                # Restores the whole state_dict, so target_proj and the four norm
                # buffers are put back exactly as well (invariant 2), whatever ran.
                model.load_state_dict(best_state)
            model.eval()

            record.update({
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "epochs_run": int(epochs_run),
                "best_epoch": int(best_epoch) if use_val else int(epochs_run),
                "selected_pretrained": bool(use_val and best_epoch == 0),
                "early_stopped": bool(use_val and epochs_run < max_epochs),
                "val_loss_pretrained": _round(history[0]) if use_val else None,
                "val_loss_best": _round(best_loss) if use_val else None,
                "val_loss_history": [_round(v) for v in history] if use_val else [],
            })
            self.fit_report_.append(record)
            if verbose:
                print("  fold %d: %d pairs (%d positive), ESS %.0f, epoch %d/%d "
                      "selected%s" % (r, n, record["positives"], record["ess"],
                                      record["best_epoch"], epochs_run,
                                      " (pretrained)" if record["selected_pretrained"] else ""),
                      flush=True)

        n_pre = sum(1 for f in self.fit_report_ if f.get("selected_pretrained"))
        self.diagnostics_["folds_selected_pretrained"] = int(n_pre)
        self.diagnostics_["max_epochs"] = int(max_epochs)
        self.diagnostics_["early_stopping"] = bool(early_stopping)
        return self

    # ---- persist ---------------------------------------------------------
    def save(self, out_dir):
        """Write a bundle in the pretrained format, so a tuned model loads through
        exactly the same path as the baseline."""
        out_dir = str(out_dir).rstrip("/\\")
        os.makedirs(out_dir, exist_ok=True)
        blob = {}
        for r, m in enumerate(self.models):
            blob.update(m.fold_state(r))
        blob.update(self.models[0].constants_state())   # shared; never changed

        # Carry forward the source bundle's provenance keys. Nothing in the load
        # path reads meta.*, so dropping them would not break anything -- it would
        # just quietly lose the record of which pretrained bundle this came from,
        # which is the one thing a saved artifact should never lose. Our own meta
        # keys are written afterwards so they win on any name clash.
        try:
            src = np.load(self.bundle_npz, allow_pickle=True)
            for k in src.files:
                if k.startswith("meta.") and k not in blob:
                    blob[k] = src[k]
        except Exception:
            pass                                        # provenance is best-effort

        blob["meta.finetuned"] = np.array(True)
        blob["meta.subspace_mode"] = np.array(
            self.subspace_.mode if self.subspace_ is not None else "none")
        path = "%s/%s" % (out_dir, BUNDLE_FILENAME)
        np.savez(path, **blob)
        return path


# ---- helpers -------------------------------------------------------------
def _snapshot(model):
    """A detached copy of every parameter and buffer.

    The whole state_dict, not just the trainable half: restoring it is then a
    guarantee about target_proj and the four normalisation buffers (invariant 2)
    rather than an assumption that nothing touched them.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _weighted_loss(model, bce, Xt, Xd, Y, W, idx):
    """Self-normalised weighted BCE on `idx`.

    Self-normalised (`sum(w*l)/sum(w)`) here rather than the plain mean used for the
    training gradient: this number is compared across epochs on a *fixed* set, so
    dividing by the set's own weight mass makes it a weighted average loss and keeps
    it on the same scale as an unweighted validation loss.
    """
    if idx.numel() == 0:
        return float("inf")
    was_training = model.training
    model.eval()
    l = bce(model(Xt[idx], Xd[idx]), Y[idx])
    out = float((W[idx] * l).sum() / W[idx].sum())
    if was_training:
        model.train()
    return out


def _stratified_split(labels, val_fraction, seed):
    """Label-stratified train/validation indices.

    Stratified because the folds are ~26% positive and a random 20% of a small fold
    can come back all-negative, at which point the validation loss is measuring the
    negative rate rather than the model. Seeded per fold (`seed + r`) so the five
    folds stay independent of one another.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if val_fraction <= 0 or n == 0:
        return np.arange(n), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    val = []
    for lab in np.unique(labels):
        pos = np.flatnonzero(labels == lab)
        rng.shuffle(pos)
        k = int(round(len(pos) * val_fraction))
        if len(pos) >= 2:
            k = max(1, min(k, len(pos) - 1))     # never take a whole class
        else:
            k = 0
        val.append(pos[:k])
    va = np.sort(np.concatenate(val)) if val else np.zeros(0, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[va] = False
    return np.flatnonzero(mask), va.astype(np.int64)


def _round(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)
