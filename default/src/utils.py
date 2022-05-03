from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


__all__ = ("get_scores",)


def get_scores(
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    """Return balanced accuracy, f1 score, roc-auc and confusion matrix"""
    return {
        "accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, y_pred, sample_weight=sample_weight),
        "roc_auc": roc_auc_score(y_true, y_prob[:, 1], sample_weight=sample_weight),
        "cm": confusion_matrix(
            y_true, y_pred, sample_weight=sample_weight, normalize="true"
        ).round(2),
    }
