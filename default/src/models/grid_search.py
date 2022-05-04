import logging

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from default.src.data import setup_features_and_labels
from default.src.utils import get_scores
from default.constants import RNG
from default.settings import N_SAMPLES, TEST_SIZE

logger = logging.getLogger(__name__)


# Settings
# ----
GRID_CONFIG = {
    #  "logistic": (
    #      LogisticRegression,
    #      {
    #          "classification__C": [0.01, 0.1, 1],
    #          "classification__penalty": ["l1", "l2"],
    #          "classification__solver": ["saga"],
    #      },
    #  ),
    "random_forest": (
        RandomForestClassifier(n_estimators=50, class_weight="balanced"),
        {
            "classification__max_depth": [3, 6, 9],
            "classification__max_features": ["sqrt", None],
        },
    ),
}


# Setup
# ----
df, X, y = setup_features_and_labels(
    kind="processed",
    n_samples=N_SAMPLES,
    random_state=RNG,
)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG
)

smote = SMOTEENN(random_state=RNG)


for name, (clf, params) in GRID_CONFIG.items():
    logger.info(f"Grid search for model {name}")
    model = Pipeline([("sampling", SMOTEENN()), ("classification", clf)])
    grid = GridSearchCV(model, params, scoring="f1", cv=5, verbose=3, n_jobs=-1)
    grid.fit(X_trainval, y_trainval)
