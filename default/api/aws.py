import logging
from io import BytesIO
from typing import Optional

import boto3
import joblib
from sklearn.base import ClassifierMixin

from default.constants import ROOT_DIR
from default.settings import (
    AWS_S3_ACCESS_KEY,
    AWS_S3_BUCKET,
    AWS_S3_DEPLOYED_MODEL_FILENAME,
    AWS_S3_REGION,
    AWS_S3_SECRET_KEY,
)


logger = logging.getLogger(__file__)


__all__ = ("create_bucket", "get_model_from_bucket", "upload_model")


# Globals
# ----
config = {
    "service_name": "s3",
    "aws_access_key_id": AWS_S3_ACCESS_KEY,
    "aws_secret_access_key": AWS_S3_SECRET_KEY,
}
s3 = boto3.resource(**config)
client = boto3.client(**config)


# AWS Helpers
# ----


def create_bucket() -> bool:
    """Util to create a new bucket as specified in `default.settings`"""
    try:
        _ = s3.create_bucket(
            Bucket=AWS_S3_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": AWS_S3_REGION},
        )
        return True
    except client.exceptions.BucketAlreadyOwnedByYou:
        logger.warning("Bucket already created")
    except client.exceptions.BucketAlreadyExists:
        logger.error("Bucket name already taken, provide another one.")
    except client.exceptions.ClientError as e:
        logger.error(f"Invalid bucket name = {e}.")

    return False


def get_model_from_bucket() -> ClassifierMixin:
    """Util to load model from bucket"""
    with BytesIO() as fp:
        s3.Bucket(AWS_S3_BUCKET).download_fileobj(AWS_S3_DEPLOYED_MODEL_FILENAME, fp)
        fp.seek(0)
        return joblib.load(fp)


def upload_model(name: Optional[str] = None) -> bool:
    """Upload a local/trained model to bucket"""
    name = name or AWS_S3_DEPLOYED_MODEL_FILENAME
    filename = name
    if not filename.endswith(".joblib"):
        filename = f"{name}.joblib"

    local_model_filepath = f"{ROOT_DIR}/models/{filename}"
    try:
        client.upload_file(Filename=local_model_filepath, Bucket=AWS_S3_BUCKET, Key=name)
        return True
    except client.exceptions.ClientError as e:
        logger.error(f"Failed to upload model - {e}")
        return False


if __name__ == "__main__":
    # Setup model
    # ----
    _ = create_bucket()
    try:
        model = get_model_from_bucket()
    except Exception as e:
        upload_model()
