from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from default.constants import RNG
from default.settings import N_SAMPLES, TEST_SIZE
from default.src.data import setup_features_and_labels


# Settings
# ----
GRID_CONFIG = {
    "gradient_boost": (
        GradientBoostingClassifier(loss="deviance", n_estimators=50, random_state=RNG),
        {
            "classification__max_depth": [20, 25, 30],
            "classification__max_features": [8, 10, 12],
            "classification__min_samples_leaf": [0, 1, 2],
            "classification__min_impurity_decrease": [0, 0.05, 0.1],
        },
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=50, random_state=RNG),
        {
            "classification__max_depth": [4, 6, 8, 10],
            "classification__max_features": [5, 10, 15, 20],
            "classification__class_weight": [{1: w} for w in (10, 15, 20, 25)],
            "classification__oob_score": [True, False],
        },
    ),
}


# Setup
# ----
df, X, y = setup_features_and_labels(kind="interim", random_state=RNG)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG
)
scaler = preprocessing.PowerTransformer()

grids = {}
for name, (clf, params) in GRID_CONFIG.items():
    print(f"Grid search for model {name}")

    pipe = Pipeline([("scaler", scaler), ("classification", clf)])
    grid = GridSearchCV(pipe, params, scoring="f1", cv=5, verbose=3, n_jobs=-1)
    grids[name] = grid

    grid.fit(X_trainval, y_trainval)

    y_pred = grid.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    rpt = classification_report(y_test, y_pred)
    print(f"---> Best model: {grid.best_params_}")
    print(f"---> Out-of-sample score: {grid.score(X_test, y_test)}")
    print(rpt)
