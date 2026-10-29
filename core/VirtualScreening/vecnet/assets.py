"""
Asset location. All paths stay explicit function/constructor arguments; this module
only offers a resolver so a caller (notably R via reticulate) can pass ONE root
instead of four paths.

Layout expected by `Assets.from_root(root)`:

    root/
      assets/bundle_baseline/vecnet_inductive.npz      <- pretrained cold-start bundle
      assets/featurization/mol2Vec_300dim.pkl         <- frozen drug encoder
      assets/featurization/protVec_100dim_3grams.csv  <- frozen target encoder
      assets/folds/train_{0..4}.csv           <- labeled splits (fine-tune only)

Any field may be overridden individually. `require(finetune=...)` validates only
what the requested run actually needs: the fold splits are required for
`finetune=True` and irrelevant otherwise.
"""
import os


class Assets:
    def __init__(self, bundle_dir, mol2vec_path, protvec_csv, folds_dir=None,
                 cache_dir=None):
        self.bundle_dir = _clean(bundle_dir)
        self.mol2vec_path = _clean(mol2vec_path)
        self.protvec_csv = _clean(protvec_csv)
        self.folds_dir = _clean(folds_dir) if folds_dir else None
        self.cache_dir = _clean(cache_dir) if cache_dir else None

    @classmethod
    def from_root(cls, root, bundle_dir=None, mol2vec_path=None, protvec_csv=None,
                  folds_dir=None, cache_dir=None):
        """Resolve the standard layout under `root`; any field can be overridden."""
        root = _clean(root)
        return cls(
            bundle_dir=bundle_dir or "%s/assets/bundle_baseline" % root,
            mol2vec_path=mol2vec_path or "%s/assets/featurization/mol2Vec_300dim.pkl" % root,
            protvec_csv=protvec_csv or "%s/assets/featurization/protVec_100dim_3grams.csv" % root,
            folds_dir=folds_dir or "%s/assets/folds" % root,
            cache_dir=cache_dir or "%s/assets/cache" % root,
        )

    def bundle_npz(self):
        return "%s/%s" % (self.bundle_dir, BUNDLE_FILENAME)

    def require(self, finetune=False):
        """Fail early, with the missing path named, before any slow work starts."""
        missing = []
        for label, path in (("pretrained bundle", self.bundle_npz()),
                            ("mol2vec encoder", self.mol2vec_path),
                            ("protvec encoder", self.protvec_csv)):
            if not os.path.exists(path):
                missing.append("%s: %s" % (label, path))
        if finetune:
            if not self.folds_dir:
                missing.append("fold splits: folds_dir not set (required when finetune=True)")
            else:
                for r in range(5):
                    p = "%s/train_%d.csv" % (self.folds_dir, r)
                    if not os.path.exists(p):
                        missing.append("fold split: %s" % p)
        if missing:
            raise FileNotFoundError(
                "missing required asset(s):\n  " + "\n  ".join(missing))
        return self


BUNDLE_FILENAME = "vecnet_inductive.npz"   # cold-start / unseen-nodes; the only scenario


def _clean(p):
    return str(p).rstrip("/\\")
