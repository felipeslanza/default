import boto3
import joblib
from flask import Flask, abort, jsonify, request

from .aws import get_model_from_bucket
from default import settings
from default.src.data import parse_features_from_dict


# ===========================================================
# Setup & hooks
# ===========================================================
app = Flask(__name__)
model = get_model_from_bucket()


@app.before_request
def force_json_payload():
    if request.method == "POST" and not request.is_json:
        abort(400, description="Requires 'Content-Type: application/json'")


# ===========================================================
# API Routes
# ===========================================================
@app.route("/api", methods=["POST"])
def index():
    processed_data = parse_features_from_dict(request.json)
    result = model.predict_proba(processed_data)

    return jsonify(result), 200
