import os

import numpy as np
import pandas as pd

from default.settings import RANDOM_STATE


# General
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Models / validation
RNG = np.random.RandomState(RANDOM_STATE)

# Data-related
DTYPE_MAPPER = {
    "categorical": "category",
    "numeric": "float64",
    "boolean": "int64",
    "text": "object",
}

FEATURES_DTYPES = (
    pd.read_csv(
        f"{DATA_DIR}/meta_features.csv",
        sep=";",
        header=None,
        index_col=0,
    )[1]
    .map(DTYPE_MAPPER)
    .to_dict()
)
