from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from default.src.data import setup_features_and_labels, train_valid_test_split
from default.src.utils import get_balanced_scores
from default.constants import RNG
from default.settings import N_SAMPLES, TEST_SIZE


df, X, y = setup_features_and_labels(kind=DATASET, n_samples=N_SAMPLES, random_state=RNG)
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(
    X, y, test_size=TEST_SIZE, random_state=RNG
)
sample_weight_train = compute_sample_weight("balanced", y_train)
sample_weight_valid = compute_sample_weight("balanced", y_valid)
