from pathlib import Path

import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from default.constants import ROOT_DIR
from default.src.data import setup_features_and_labels


if __name__ == "__main__":
    # Settings
    # ----
    MODEL_NAME = Path(__file__).stem
    MODEL_CLASS = RandomForestClassifier
    MODEL_KWARGS = {
        "n_estimators": 1000,
        "max_depth": 6,
        "max_features": 15,
        "oob_score": False,
        "class_weight": {1: 15},
    }

    # Setup
    # ----
    df, X, y = setup_features_and_labels(kind="interim")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = MODEL_CLASS(**MODEL_KWARGS)
    scaler = preprocessing.StandardScaler()

    clf = Pipeline([("scaler", scaler), ("classification", model)])
    clf.name = MODEL_NAME
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)

    rpt = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"---> Loaded model: {MODEL_NAME}")
    print(rpt)

    # Persist model
    # ----
    joblib.dump(clf, f"{ROOT_DIR}/models/{MODEL_NAME}.joblib")
