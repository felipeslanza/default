import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from default.src.data import load_dataset
from default.src.utils import get_balanced_scores, train_valid_test_split


logger = logging.getLogger(__name__)


def setup_data(n_samples: int) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_dataset("preprocessed")
    df = df.loc[df.default.notnull()].sample(n_samples)
    X = df.drop("default", axis=1).fillna(-1).to_numpy()
    y = df.default.to_numpy()

    return df, X, y


# Settings
# ========

# General
N_SAMPLES = 40_000
TEST_SIZE = 0.3
RANDOM_STATE = 222

# Decision
PROBABILITY_CUTOFF_OPTIONS = (0.01, 0.1, 0.25, 0.5)

CLASSIFIERS = {
    "logistic": LogisticRegression(C=0.01, penalty="l2", solver="saga"),
    "knn": KNeighborsClassifier(4),
    "naive_bayes": GaussianNB(),
    # "neural_net": MLPClassifier(
    #     alpha=0.1, solver="lbfgs", max_iter=10_000, learning_rate_init=0.001, shuffle=True
    # ),
    "decision_tree": DecisionTreeClassifier(max_features=5),
    "random_forest": RandomForestClassifier(
        max_depth=None, n_estimators=100, max_features="log2"
    ),
}


# Setup
# =====
n_models = len(CLASSIFIERS)
n_cutoffs = len(PROBABILITY_CUTOFF_OPTIONS)

df, X, y = setup_data(N_SAMPLES)
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
sample_weight_train = compute_sample_weight("balanced", y_train)
sample_weight_valid = compute_sample_weight("balanced", y_valid)

acc = np.zeros((n_models, n_cutoffs), dtype=np.float64)
f1 = np.zeros((n_models, n_cutoffs), dtype=np.float64)
roc_auc = np.zeros((n_models, 1), dtype=np.float64)
cm = np.zeros((n_models, n_cutoffs, 2, 2), dtype=np.float16)

for i, (name, model) in enumerate(CLASSIFIERS.items()):
    logger.info(f"Traing model [{name}]")
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
    except TypeError:
        model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_valid)
    for j, prob_cutoff in enumerate(PROBABILITY_CUTOFF_OPTIONS):
        y_pred = (y_prob[:, 1] > prob_cutoff).astype(int)
        acc[i, j], f1[i, j], roc_auc[i], cm[i, j] = get_balanced_scores(
            y_pred, y_prob, y_valid, sample_weight_valid
        ).values()
