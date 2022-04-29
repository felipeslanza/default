from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# Settings
# ========
N_SPLITS = 10
RANDOM_STATE = 222


# Setup
# =====
df = load_dataset("interim")
valid = df.default.notnull()
X = df.loc[valid].drop("default", axis=1).to_numpy()
y = df[valid, "default"].to_numpy()

classifiers = {
    "gaussian": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "svm_linear": SVC(kernel="linear", C=0.025),
    "svm_rbf": SVC(kernel="rbf", gamma="scale", C=1),
    "neural_net": MLPClassifier(
        alpha=1,
        max_iter=10_000,
        learning_rate_init=0.001,
        shuffle=True,
    ),
    "knn": KNeighborsClassifier(3),
}
