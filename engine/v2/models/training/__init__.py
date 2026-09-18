"""Model training: current dataset/training recipes, folds, receipts and the job.

Layer 6 of `system_rearchitecture.md` §4.1. Replaces `models/training/,
rewritten above scoring rather than moved`. P5-3 extracts the legacy recipes
as data (:mod:`.recipes`), reproduces their membership and folds
(:mod:`.folds`), writes per-fold membership and label-availability receipts
(:mod:`.receipts`) and fits them only through the explicit, resumable job
(:mod:`.job`). See ``README.md``.
"""

from .estimators import (
    EqualWeightBlend,
    LogTargetModel,
    SeedMeanEnsemble,
    UnsupportedEstimator,
    fit_recipe_estimator,
)
from .fold_store import FoldOutcome, TrainingJobResult, TrainingRefused
from .folds import (
    FoldPlan,
    PreparedDataset,
    RecipeDataError,
    dataset_fingerprint,
    plan_folds,
    prepare_dataset,
)
from .job import TRAINING_JOB_V1, run_training_job
from .receipts import (
    LABEL_RECEIPT_V1,
    MEMBERSHIP_RECEIPT_V1,
    ReceiptIssue,
    fold_receipts,
    receipt_issues,
)
from .recipes import (
    CLOCK_ID,
    LEGACY_SEED,
    OWNER_P5_4,
    OWNER_TRAINING_JOB,
    RECIPE_V1,
    EstimatorSpec,
    FoldScheme,
    LabelRule,
    RecipeKey,
    ResidualRule,
    RowFilter,
    TargetSpec,
    ThresholdRule,
    TrainingRecipe,
    UpstreamDependency,
    ValueMask,
    current_recipes,
    recipe_fingerprint,
)

__all__ = [
    "CLOCK_ID", "EqualWeightBlend", "EstimatorSpec", "FoldOutcome", "FoldPlan", "FoldScheme",
    "LABEL_RECEIPT_V1", "LEGACY_SEED", "LabelRule", "LogTargetModel", "MEMBERSHIP_RECEIPT_V1",
    "OWNER_P5_4", "OWNER_TRAINING_JOB", "PreparedDataset", "RECIPE_V1", "ReceiptIssue",
    "RecipeDataError", "RecipeKey", "ResidualRule", "RowFilter", "SeedMeanEnsemble",
    "TRAINING_JOB_V1", "TargetSpec", "ThresholdRule", "TrainingJobResult", "TrainingRecipe",
    "TrainingRefused", "UnsupportedEstimator", "UpstreamDependency", "ValueMask",
    "current_recipes", "dataset_fingerprint", "fit_recipe_estimator", "fold_receipts",
    "plan_folds", "prepare_dataset", "receipt_issues", "recipe_fingerprint", "run_training_job",
]
