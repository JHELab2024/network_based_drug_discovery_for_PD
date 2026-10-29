"""
Inference: 5-fold-averaged P(bind) for a drug library against one or many targets.

Only the *source* of the drug vectors changes relative to the verified baseline
(cache instead of re-featurizing). The arithmetic is deliberately identical:

  - same dtypes (raw float64 in, normalization in float64 inside the model,
    float32 through the linear stack);
  - same batch boundaries (`batch_size`, last batch short) -- a matmul's reduction
    order can depend on its M dimension, so the batching is preserved rather than
    "optimized", which is what keeps parity exact;
  - same fold accumulation order and the same divide-by-5 at the end;
  - unparseable SMILES -> nan, never dropped from the output shape.

Cost note: the matmul is ~0.3 s/target for a 0.52M library on a 3060 -- three orders
of magnitude under featurization, so there is nothing to gain by fusing it and a
parity guarantee to lose.

A3 addition (`spread_out` / `spread`): the five fold probabilities are already
computed here and then averaged away, so their per-compound spread -- the fold
disagreement diagnostic -- costs one extra device-to-host copy per fold and nothing
else. It is an optional *out* parameter rather than a second return value so the
parity-critical path above is byte-for-byte what it was: the accumulation, the
divide-by-5 and the dtypes are untouched, and with `spread_out=None` the only added
work is one `is not None` test per fold.
"""
import numpy as np
import torch

from .model import VecNet

N_FOLDS = 5


def load_folds(bundle_npz, device):
    """Pretrained-format npz -> 5 eval-mode fold models on `device`."""
    z = np.load(bundle_npz, allow_pickle=True)
    return [VecNet().to(device).eval().load_bundle(z, r) for r in range(N_FOLDS)]


@torch.no_grad()
def score_one_target(models, raw_drug, valid, raw_target, device, batch_size=4096,
                     spread_out=None):
    """(N,300) raw drug + (100,) raw target -> (N,) P(bind), nan where invalid.

    spread_out: optional (N,) float array. When given, it is filled with the
    population standard deviation of the five folds' probabilities per compound --
    A3's fold-disagreement diagnostic. Compounds that are invalid, or all of them
    when `rows` is empty, are left as they were (nan).
    """
    result = np.full(raw_drug.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return result

    drug = raw_drug[rows]
    n = rows.size
    total = torch.zeros(n, 1, dtype=torch.float32, device=device)
    per_fold = (np.empty((n, len(models)), dtype=np.float64)
                if spread_out is not None else None)
    for k, m in enumerate(models):
        was_training = m.training
        m.eval()
        fold = torch.zeros(n, 1, dtype=torch.float32, device=device)
        for s in range(0, n, batch_size):
            sl = slice(s, min(s + batch_size, n))
            xd = torch.tensor(drug[sl], dtype=torch.float64, device=device)
            xt = torch.tensor(np.repeat(raw_target[None, :], sl.stop - sl.start, axis=0),
                              dtype=torch.float64, device=device)
            fold[sl] = torch.sigmoid(m(xt, xd))
        total += fold
        if per_fold is not None:
            per_fold[:, k] = fold.cpu().numpy().reshape(-1)
        if was_training:
            m.train()
    result[rows] = (total / len(models)).cpu().numpy().reshape(-1)
    if per_fold is not None:
        spread_out[rows] = per_fold.std(axis=1)
    return result


@torch.no_grad()
def score_targets(models, raw_drug, valid, raw_targets, device, batch_size=4096,
                  verbose=False, spread=None):
    """Many targets -> (N_drugs, N_targets) P(bind).

    The drug vectors are featurized once by the caller and reused for every target;
    this loop is what turns "0.52M x 20 targets" from hours into minutes.

    spread: optional (N_drugs, N_targets) array, filled column by column with the
    across-fold standard deviation (see `score_one_target`).
    """
    out = np.full((raw_drug.shape[0], len(raw_targets)), np.nan, dtype=np.float64)
    for j, rt in enumerate(raw_targets):
        if verbose:
            print("  scoring target %d/%d" % (j + 1, len(raw_targets)), flush=True)
        col = None if spread is None else np.full(raw_drug.shape[0], np.nan, dtype=np.float64)
        out[:, j] = score_one_target(models, raw_drug, valid, rt, device, batch_size,
                                     spread_out=col)
        if spread is not None:
            spread[:, j] = col
    return out
