from flask import Flask, abort, jsonify, request

from .aws import get_model_from_bucket
from default.src.data import parse_features_from_dict


__all__ = ("app", "model")


# ===========================================================
# Setup & hooks
# ===========================================================
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
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
    df = parse_features_from_dict(raw)
    X = df.round(2).to_numpy()
    y_prob = model.predict_proba(X)
    obj = {"uuid": raw["uuid"], "prob": y_prob[0][1].round(2)}

    return jsonify(obj), 200
