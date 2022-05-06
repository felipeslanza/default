from flask import Flask, abort, jsonify, request

from .aws import get_model_from_bucket
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
    raw = request.json
    X = parse_features_from_dict(raw)
    ### TEMP ###
    breakpoint()
    ### TEMP ###
    y_prob = model.predict_proba(X)
    obj = {"uuid": raw["uuid"], "prob": y_prob[0][1]}

    return jsonify(obj), 200
