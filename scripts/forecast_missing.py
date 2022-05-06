import joblib

from default.constants import ROOT_DIR
from default.src.data import load_dataset, setup_features_and_labels


# Settings
# ----
MODEL_FILENAME = "random_forest_v1"
DATASET = "raw"


if __name__ == "__main__":
    # Setup
    # ----
    clf = joblib.load(f"{ROOT_DIR}/models/{MODEL_FILENAME}.joblib")

    df, X, y = setup_features_and_labels(kind="interim", drop_missing=False, fillna_X=-1)
    missing = df.default.isna()
    X = X[missing]

    raw_df = load_dataset("raw")
    missing_uuids = raw_df.loc[missing, "uuid"]

    # Forecast
    # ----
    y_prob = clf.predict_proba(X)

    final = pd.DataFrame({"uuid": missing_uuids, "pd": y_prob[:, 1].round(2)})
    final.to_csv(f"{ROOT_DIR}/data/output/missing.csv", sep=";", index=False)
