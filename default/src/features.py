import logging

from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
    SelectPercentile,
    SelectKBest,
)

from default.src.data import setup_features_and_labels, train_valid_test_split

logger = logging.getLogger(__name__)

# Settings
# ----
SCORER_COMBOS = [
    (SelectPercentile, {"score_func": f_classif, "percentile": 25}),
    (SelectPercentile, {"score_func": mutual_info_classif, "percentile": 25}),
    (SelectKBest, {"score_func": f_classif, "k": 10}),
    (SelectKBest, {"score_func": mutual_info_classif, "k": 10}),
]


if __name__ == "__main__":
    from default.constants import DATA_DIR, RNG
    from default.settings import TEST_SIZE

    # Setup
    # ----
    df, X, y = setup_features_and_labels()
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
        X, y, test_size=TEST_SIZE, random_state=RNG
    )

    # Selection
    # ----
    features = df.columns[1:]
    selected_features = set()
    for (selector_cls, kwargs) in SCORER_COMBOS:
        scorer = kwargs["score_func"].__name__
        thresh = kwargs.get("percentile") or kwargs.get("k")
        logger.info(f"\n---> {selector_cls.__name__}: {scorer} @ {thresh}")

        logger.info(f"---> Pre: {X_train.shape}")
        selector = selector_cls(**kwargs)
        X_train_new = selector.fit_transform(X_train, y_train)
        logger.info(f"---> Post: {X_train_new.shape}")

        feats = selector.get_feature_names_out(features)
        selected_features.update(feats)

    print(sorted(selected_features))

    df = df[["default", *[i for i in df if i in selected_features]]]  # preserving order
    df.to_csv(f"{DATA_DIR}/processed.csv", sep=";", index=False)
