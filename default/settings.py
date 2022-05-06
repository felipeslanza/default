import os


# General
# ----
DEBUG = True
LOGGING_LEVEL = "INFO"
LOGGING_FORMAT = f"%(asctime)s - %(levelname)s - %(module)s - %(message)s"


# API
# ----
AWS_S3_BUCKET = "default-models-storage"
AWS_S3_ACCESS_KEY = os.environ.get("AWS_S3_ACCESS_KEY")
AWS_S3_SECRET_KEY = os.environ.get("AWS_S3_SECRET_KEY")
AWS_S3_REGION = "sa-east-1"
AWS_S3_DEPLOYED_MODEL_FILENAME = "random_forest_v1"


# Models / validation
# ----
N_SAMPLES = 20_000
RANDOM_STATE = 222
TEST_SIZE = 0.2  # hold-out samples to be used for final model evaluation
MAX_MISSING_FEATURE_PCT = 0.7  # Features above this threshold will be dropped
