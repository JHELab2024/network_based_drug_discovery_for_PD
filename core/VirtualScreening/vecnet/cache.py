"""
Drug-embedding cache.

Featurization -- not the matmul -- is the screening bottleneck: measured ~3.3k
drugs/s single-threaded, so a 0.52M library costs ~2.7 min *per target* if
re-featurized. Screening 20 targets that way is ~0.9 h of repeated identical work.
Here every unique drug is featurized once and the raw 300d vector reused for every
target and every fold.

Two layers:
  - in-memory, for the life of the object -> multi-target reuse in one screen();
  - on-disk (optional), keyed by a content hash of the unique SMILES list ->
    reuse across sessions and across baseline/tuned models.

The cached vectors are the **raw** (un-normalized) float64 sums. The featurizer and
the four normalization constants are frozen, so a raw vector is valid for the
baseline and for every fine-tuned bundle alike. float64 is deliberate: the model
normalizes in float64 before casting to float32, so caching float32 here would
perturb that path and break bit-parity with the baseline.

Unparseable SMILES are stored as all-NaN rows and reported via a boolean mask.
"""
import hashlib
import os

import numpy as np

DRUG_DIM = 300

# Bump whenever the meaning of a cached vector changes (featurization or validity
# rules). The tag is hashed into the key, so stale arrays become unreachable rather
# than being served as if current -- a silently wrong screen is the worst failure
# this cache can produce, and it has no symptom.
#   v1 -> initial
#   v2 -> empty (0-atom) molecules are invalid (nan) instead of an all-zero vector
CACHE_VERSION = "v2"


def library_key(unique_smiles, version=CACHE_VERSION):
    """Content hash pinning the exact unique-SMILES list, its order, and the
    featurization semantics.

    Because the key pins the order, the cached array's row i always corresponds to
    unique_smiles[i] -- so the SMILES themselves never need storing.
    """
    h = hashlib.md5()
    h.update(("vecnet-drugcache-%s\n" % version).encode("utf-8"))
    for s in unique_smiles:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


class DrugEmbeddingCache:
    """Featurize unique drugs once; serve raw 300d vectors."""

    def __init__(self, featurizer, cache_dir=None):
        """cache_dir None/False -> in-memory only (still de-duplicates and still
        reuses across targets within a run); a path -> also persist across runs."""
        self.feat = featurizer
        self.cache_dir = str(cache_dir).rstrip("/\\") if cache_dir else None
        self._mem = {}   # smiles -> 300d float64 or None

    # ---- internals -------------------------------------------------------
    def _raw(self, smiles):
        """Raw 300d vector, or None if `smiles` is not a molecule.

        Guards an RDKit sharp edge the frozen featurizer inherits: MolFromSmiles("")
        returns an empty **0-atom molecule** rather than None, so a blank SMILES cell
        -- common in user-supplied library CSVs -- would otherwise featurize to an
        all-zero vector and be scored like a real compound, yielding a confident but
        meaningless number instead of nan.

        Two cheap guards, no second RDKit parse (featurization is the bottleneck):
          - blank/whitespace strings never reach the featurizer;
          - an exactly all-zero sum means the molecule produced no Morgan identifiers,
            i.e. it had no atoms. A real compound has >=1 identifier, so its 300
            float64 components cannot all cancel to exactly 0.
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        v = self.feat.drug_raw(smiles)
        if v is None or not v.any():
            return None
        return v

    def _featurize_unique(self, unique_smiles, verbose=False):
        out = np.full((len(unique_smiles), DRUG_DIM), np.nan, dtype=np.float64)
        step = max(1, len(unique_smiles) // 10)
        for i, smi in enumerate(unique_smiles):
            v = self._mem.get(smi, False)
            if v is False:
                v = self._raw(smi)
                self._mem[smi] = v
            if v is not None:
                out[i] = v
            if verbose and (i + 1) % step == 0:
                print("  featurized %d/%d drugs" % (i + 1, len(unique_smiles)), flush=True)
        return out

    def _disk_path(self, key):
        return "%s/drugs_%s.npy" % (self.cache_dir, key)

    # ---- public ----------------------------------------------------------
    def encode(self, smiles_list, verbose=False):
        """SMILES list -> (raw (N,300) float64, valid (N,) bool).

        Duplicates in `smiles_list` cost nothing: each unique string is featurized
        once and broadcast back to every position it occupies.
        """
        unique = list(dict.fromkeys(smiles_list))       # stable, de-duplicated
        table = None

        if self.cache_dir:
            key = library_key(unique)
            path = self._disk_path(key)
            if os.path.exists(path):
                cand = np.load(path)
                if cand.shape == (len(unique), DRUG_DIM):   # hash pins identity
                    table = cand
                    if verbose:
                        print("  drug cache hit: %s (%d unique)" % (os.path.basename(path), len(unique)), flush=True)
            if table is None:
                if verbose:
                    print("  drug cache miss: featurizing %d unique drugs" % len(unique), flush=True)
                table = self._featurize_unique(unique, verbose=verbose)
                os.makedirs(self.cache_dir, exist_ok=True)
                tmp = path + ".tmp%d.npy" % os.getpid()     # atomic: never a torn cache
                np.save(tmp, table)
                os.replace(tmp, path)
        else:
            table = self._featurize_unique(unique, verbose=verbose)

        index = {s: i for i, s in enumerate(unique)}
        rows = np.fromiter((index[s] for s in smiles_list), dtype=np.int64,
                           count=len(smiles_list))
        raw = table[rows]
        valid = ~np.isnan(raw).any(axis=1)
        return raw, valid
