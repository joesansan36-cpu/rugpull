import json
import os
import re
import time
from pathlib import Path

import numpy as np
import umap
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer


MAX_CONTENT_LENGTH = 100 * 1024 * 1024
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache")).resolve()
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
EMBEDDING_DIMENSIONS = 384

app = Flask(__name__, static_folder="./static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

model = SentenceTransformer("all-MiniLM-L12-v2", device="cpu")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def safe_cache_name(value):
    if not isinstance(value, str):
        raise ValueError("cache name must be a string")
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("cache name may contain only letters, numbers, hyphens, and underscores")
    if not value or value in {".", ".."}:
        raise ValueError("invalid cache name")
    return value


def cache_directory(name):
    name = safe_cache_name(name)
    directory = (CACHE_DIR / name).resolve()
    if CACHE_DIR not in directory.parents:
        raise ValueError("invalid cache path")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def binary_float_matrix(data, columns):
    if not data:
        raise ValueError("empty binary payload")
    if len(data) % (columns * np.dtype(np.float32).itemsize) != 0:
        raise ValueError(f"payload must contain complete {columns}-float rows")
    array = np.frombuffer(data, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("payload contains invalid numeric values")
    return array.reshape((-1, columns))


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_file(path):
    return send_from_directory(app.static_folder, path)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/post", methods=["POST"])
def post():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or not data:
        return jsonify({"error": "JSON body must be a non-empty array of strings"}), 400
    if len(data) > 5000:
        return jsonify({"error": "maximum 5000 strings per request"}), 400
    if any(not isinstance(item, str) or len(item) > 10000 for item in data):
        return jsonify({"error": "each item must be a string of at most 10000 characters"}), 400

    encoded = model.encode(
        data,
        show_progress_bar=False,
        device="cpu",
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if encoded.shape != (len(data), EMBEDDING_DIMENSIONS):
        return jsonify({"error": "unexpected embedding dimensions"}), 500
    return Response(encoded.astype(np.float32).tobytes(), mimetype="application/octet-stream")


@app.route("/list-files", methods=["POST"])
def list_post_files():
    directories = sorted(
        directory.name for directory in CACHE_DIR.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    return jsonify(directories)


@app.route("/post-umap", methods=["POST"])
def post_umap():
    try:
        matrix = binary_float_matrix(request.data, EMBEDDING_DIMENSIONS)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    row_count = matrix.shape[0]
    if row_count < 2:
        return jsonify({"error": "at least 2 rows are required"}), 400
    if row_count > 100000:
        return jsonify({"error": "maximum 100000 rows"}), 400

    start_time = time.time()
    reducer = umap.UMAP(
        n_neighbors=min(15, row_count - 1),
        min_dist=0.0,
        n_components=2,
        n_jobs=1,
        low_memory=True,
        metric="cosine",
        verbose=False,
        random_state=42,
    )
    umap_data = reducer.fit_transform(matrix)
    app.logger.info("UMAP took %.3f seconds", time.time() - start_time)
    return Response(umap_data.astype(np.float32).tobytes(), mimetype="application/octet-stream")


@app.route("/save", methods=["POST"])
def save():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return jsonify({"error": "JSON body must contain a data object"}), 400

    try:
        directory = cache_directory(payload.get("filename"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    index_path = directory / "index.json"
    with index_path.open("w", encoding="utf-8") as file:
        json.dump(payload["data"], file, ensure_ascii=False)
    return Response("Saved", mimetype="text/plain")


def save_embedding_upload(filename_field, columns, dtype, output_name):
    upload = request.files.get("file")
    if upload is None:
        return Response("No file", status=400, mimetype="text/plain")

    try:
        directory = cache_directory(request.form.get("path", ""))
        data = upload.read(MAX_CONTENT_LENGTH + 1)
        if len(data) > MAX_CONTENT_LENGTH:
            return Response("File too large", status=413, mimetype="text/plain")

        if dtype == np.float32:
            matrix = binary_float_matrix(data, columns)
        else:
            if len(data) % np.dtype(dtype).itemsize:
                raise ValueError("invalid binary payload")
            array = np.frombuffer(data, dtype=dtype)
            if len(array) % columns:
                raise ValueError(f"payload must contain complete {columns}-value rows")
            matrix = array.reshape((-1, columns))

        with (directory / output_name).open("wb") as file:
            np.save(file, matrix, allow_pickle=False)
        return Response("Saved", mimetype="text/plain")
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/save-embeddings-u8", methods=["POST"])
def save_embeddings_u8():
    return save_embedding_upload("file", EMBEDDING_DIMENSIONS, np.uint8, "embedding_u8.npy")


@app.route("/save-embeddings-ctx", methods=["POST"])
def save_embeddings_ctx():
    return save_embedding_upload("file", 2, np.float32, "embedding_ctx.npy")


@app.route("/save-results", methods=["POST"])
def save_results():
    return save_embedding_upload("file", 3, np.float32, "results.npy")


@app.route("/load", methods=["POST"])
def load():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid JSON body"}), 400
    try:
        directory = cache_directory(payload.get("filename"))
        with (directory / "index.json").open("r", encoding="utf-8") as file:
            return jsonify(json.load(file))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 404


def load_embedding(name, dtype):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid JSON body"}), 400
    try:
        directory = cache_directory(payload.get("filename"))
        with (directory / name).open("rb") as file:
            array = np.load(file, allow_pickle=False)
        if array.dtype != dtype:
            array = array.astype(dtype)
        return Response(array.tobytes(), mimetype="application/octet-stream")
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404


@app.route("/load-embeddings-u8", methods=["POST"])
def load_embeddings_u8():
    return load_embedding("embedding_u8.npy", np.uint8)


@app.route("/load-embeddings-ctx", methods=["POST"])
def load_embeddings_ctx():
    return load_embedding("embedding_ctx.npy", np.float32)


@app.route("/load-results", methods=["POST"])
def load_results():
    return load_embedding("results.npy", np.float32)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "request too large"}), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "3000")),
        debug=False,
    )
