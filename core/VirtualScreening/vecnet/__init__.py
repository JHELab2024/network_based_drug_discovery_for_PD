"""
vecnet -- the virtual-screening (AI) component: AI-Bind VecNet drug-target binding,
ported to PyTorch, with an optional fine-tune to a user's compound library.

    from vecnet import screen
    out = screen(targets=protein_seq, library=smiles_list,
                 assets_root="/.../VirtualScreeningArm",
                 device="cuda", finetune=False)
    out["scores"]     # (n_compounds, n_targets) P(bind)

Four pieces, each usable on its own:

  api / predict        screening, baseline or adapted (tier 0a / 0b)
  trainset / finetune  the adaptation: importance weighting under covariate shift,
                       weighted-validation early stopping, and the diagnostics that
                       decide before the run whether adaptation can help at all
  calibrate            library percentile + heavy-atom residual -- what makes a
                       score comparable across targets and free of the size artifact
  evaluate             scaffold split, property-matched decoys, EF / BEDROC -- the
                       held-out metric every claim about the component has to pass

`calibrate` and `evaluate` deliberately import neither torch nor scikit-learn: both
must run on a saved score table, on a machine that has neither, long after the
screen. `scikit-learn` is imported lazily inside `trainset`, so the baseline
(tier 0a) path keeps exactly the dependencies it always had.

Importable modules only: no CLI, no argparse, no config files, no import-time side
effects. Every path is an explicit argument.
"""
from .api import (
    VecNetScreener,
    normalize_library,
    normalize_targets,
    read_fasta,
    resolve_tier,
    screen,
    top_hits,
)
from .assets import Assets
from .cache import DrugEmbeddingCache
from .reference import (
    known_binders,
    reference_binders,
    reference_targets,
)
from .calibrate import (
    calibrate_matrix,
    calibrate_scores,
    heavy_atom_counts,
    library_percentile,
    size_residual,
    spearman,
)
from .device import cuda_available, describe, resolve_device
from .evaluate import (
    bedroc,
    enrichment_factor,
    enrichment_report,
    murcko_scaffold,
    per_target_enrichment,
    property_matched_decoys,
    roc_auc,
    scaffold_split,
)
from .featurizer import Featurizer
from .finetune import VecNetFineTuner
from .model import VecNet
from .predict import load_folds, score_one_target, score_targets
from .trainset import (
    ChemicalSubspace,
    build_subspace,
    effective_sample_size,
    extract_fold_pairs,
    load_fold_compounds,
    normalise_weights,
)

__all__ = [
    "reference_targets",
    "reference_binders",
    "known_binders",
    # screening
    "screen", "VecNetScreener", "top_hits", "read_fasta", "resolve_tier",
    "normalize_targets", "normalize_library",
    "Assets", "Featurizer", "VecNet", "VecNetFineTuner", "DrugEmbeddingCache",
    "load_folds", "score_one_target", "score_targets",
    "resolve_device", "cuda_available", "describe",
    # A1 / A3: subspace, weights, diagnostics
    "ChemicalSubspace", "build_subspace", "extract_fold_pairs",
    "load_fold_compounds", "normalise_weights", "effective_sample_size",
    # A2: calibration
    "calibrate_scores", "calibrate_matrix", "library_percentile", "size_residual",
    "heavy_atom_counts", "spearman",
    # A5: evaluation
    "scaffold_split", "murcko_scaffold", "property_matched_decoys",
    "enrichment_report", "per_target_enrichment", "enrichment_factor", "bedroc",
    "roc_auc",
]
