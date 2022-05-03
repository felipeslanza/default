import logging

import numpy as np
from imblearn.over_sampling import SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from default.src.data import setup_features_and_labels, train_valid_test_split
from default.src.utils import get_scores
from default.constants import RNG
from default.settings import N_SAMPLES, TEST_SIZE

logger = logging.getLogger(__name__)


# Settings
# ========
DATASET = "processed"
N_SPLITS = 5

CLASSIFIERS = {
    "logistic": LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="saga",
        random_state=RNG,
    ),
    # "knn": KNeighborsClassifier(4),
    # "naive_bayes": GaussianNB(),
    # "neural_net": MLPClassifier(
    #     alpha=0.1,
    #     solver="lbfgs",
    #     max_iter=10_000,
    #     learning_rate_init=0.001,
    #     random_state=RNG,
    # ),
    # "decision_tree": DecisionTreeClassifier(max_features=5, random_state=RNG),
    "random_forest": RandomForestClassifier(
        max_features="log2",
        max_depth=3,
        # class_weight="balanced",
        random_state=RNG,
    ),
    "gradient_boost": GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        random_state=RNG,
    ),
}
CLASSIFIERS["lr_gb_rf"] = VotingClassifier(
    estimators=[
        ("lr", CLASSIFIERS["logistic"]),
        ("gb", CLASSIFIERS["gradient_boost"]),
        ("rf", CLASSIFIERS["random_forest"]),
    ],
    voting="soft",
)


# Setup
# =====
n_models = len(CLASSIFIERS)
acc = np.zeros((N_SPLITS, n_models), dtype=np.float64)
f1 = np.zeros((N_SPLITS, n_models), dtype=np.float64)
roc_auc = np.zeros((N_SPLITS, n_models), dtype=np.float64)
cm = np.zeros((N_SPLITS, n_models, 2, 2), dtype=np.float16)

df, X, y = setup_features_and_labels(
    kind=DATASET,
    n_samples=N_SAMPLES,
    random_state=RNG,
)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG
)

smote = SMOTEENN(random_state=RNG)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
for i, (train_idx, valid_idx) in enumerate(kfold.split(X_trainval, y_trainval)):
    logger.info(f"K-Fold iteration: {i+1}/{N_SPLITS}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    X_train, y_train = smote.fit_resample(X_train, y_train)

    for j, (name, model) in enumerate(CLASSIFIERS.items()):
        logger.info(f"Training model {name}")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_valid)
        y_pred = model.predict(X_valid)

        acc[i, j], f1[i, j], _, roc_auc[i], cm[i, j] = get_scores(
            y_pred, y_prob, y_valid
        ).values()

acc = pd.DataFrame(acc, columns=CLASSIFIERS)
f1 = pd.DataFrame(f1, columns=CLASSIFIERS)
ra = pd.DataFrame(roc_auc, columns=CLASSIFIERS)
