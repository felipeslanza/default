import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


__all__ = ("get_balanced_scores", "train_test_split")


def get_balanced_scores(
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    y_true: np.ndarray,
    sample_weight: np.ndarray,
) -> dict:
    """Return balanced accuracy, f1 score, roc-auc and confusion matrix"""
    return {
        "accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
        "roc_auc": roc_auc_score(y_true, y_prob[:, 1], sample_weight=sample_weight),
        "cm": confusion_matrix(
            y_true, y_pred, sample_weight=sample_weight, normalize="true"
        ).round(2),
    }


def train_valid_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray]:
    """Utility to split in train, valid and test sets"""
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rem, y_rem, test_size=test_size / 2, random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test
