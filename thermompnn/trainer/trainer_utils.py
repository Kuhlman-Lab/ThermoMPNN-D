from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef
from torchmetrics.functional import r2_score, mean_squared_error, pearson_corrcoef


def get_metrics():
    """Torchmetrics collection for logging during training.
    NOTE: PearsonCorrCoef does not work correctly at this time."""
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


def get_metrics_functional():
    """Used for more granular metric tracking (v2, deprecated)"""
    return {
        "r2": r2_score, 
        "mse": mean_squared_error,
        "rmse": mean_squared_error,
        "pearson": pearson_corrcoef
    }