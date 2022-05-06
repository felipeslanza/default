import logging

import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from default.constants import RNG
from default.settings import N_SAMPLES, TEST_SIZE
from default.src.data import setup_features_and_labels, train_valid_test_split
from default.src.utils import get_scores

logger = logging.getLogger(__name__)


# Settings
# ========
DATASET = "interim"
# DATASET = "processed"
N_SPLITS = 5

CLASSIFIERS = {
    # "logistic": LogisticRegression(
    #     C=1.0,
    #     penalty="l2",
    #     solver="saga",
    #     random_state=RNG,
    # ),
    # "knn": KNeighborsClassifier(4),
    # "naive_bayes": GaussianNB(),
    #  "neural_net": MLPClassifier(
    #      alpha=0.1,
    #      solver="lbfgs",
    #      max_iter=10_000,
    #      learning_rate_init=0.001,
    #      random_state=RNG,
    #  ),
    # "decision_tree": DecisionTreeClassifier(max_features=5, random_state=RNG),
    "random_forest": RandomForestClassifier(
        max_depth=6,
        max_features=15,
        class_weight={1: 15},
        random_state=RNG,
    ),
    "gradient_boost": GradientBoostingClassifier(
        max_depth=25,
        max_features=10,
        random_state=RNG,
    ),
}
#  CLASSIFIERS["gb_rf"] = VotingClassifier(
#      estimators=[
#          ("gb", CLASSIFIERS["gradient_boost"]),
#          ("rf", CLASSIFIERS["random_forest"]),
#      ],
#      voting="soft",
#  )


# Setup
# =====
n_models = len(CLASSIFIERS)
acc = np.zeros((N_SPLITS, n_models), dtype=np.float64)
f1 = np.zeros((N_SPLITS, n_models), dtype=np.float64)
pr_auc = np.zeros((N_SPLITS, n_models), dtype=np.float64)
cm = np.zeros((N_SPLITS, n_models, 2, 2), dtype=np.float16)

df, X, y = setup_features_and_labels(
    kind=DATASET,
    # n_samples=N_SAMPLES,
    random_state=RNG,
)
X_trainval, _, y_trainval, _ = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG
)

scaler = preprocessing.PowerTransformer()
smote = SMOTEENN(random_state=RNG)

kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG)
for i, (train_idx, valid_idx) in enumerate(kfold.split(X_trainval, y_trainval)):
    logger.info(f"K-Fold iteration: {i+1}/{N_SPLITS}")
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    # X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    for j, (name, model) in enumerate(CLASSIFIERS.items()):
        logger.info(f"Training model {name}")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_valid)
        y_pred = model.predict(X_valid)

        acc[i, j], f1[i, j], pr_auc[i], _, cm[i, j] = get_scores(
            y_pred, y_prob, y_valid
        ).values()

acc = pd.DataFrame(acc, columns=CLASSIFIERS)
f1 = pd.DataFrame(f1, columns=CLASSIFIERS)
auc = pd.DataFrame(pr_auc, columns=CLASSIFIERS)
