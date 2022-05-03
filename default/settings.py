# General
DEBUG = True
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = f"%(asctime)s - %(levelname)s - %(module)s - %(message)s"


# Models / validation
N_SAMPLES = 50_000
RANDOM_STATE = 222
TEST_SIZE = 0.3

# Data
MAX_MISSING_FEATURE_PCT = 0.4  # Features above this threshold will be dropped
