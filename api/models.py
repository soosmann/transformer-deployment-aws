"""Load trained model artifacts on first use, then reuse them in this process."""

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from api.s3 import download_dir_from_s3

MODEL_TASKS = {
    "tinybert-sentiment-analysis": "sentiment-analysis",
    "tinybert-disaster-tweets": "text-classification",
    "vit-human-pose-classification": "image-classification",
}
_load_lock = Lock()


@lru_cache(maxsize=3)
def _load_model(model_name):
    local_path = Path(os.getenv("MODEL_DIR", "ml-model-dir")) / model_name
    if not local_path.is_dir():
        prefix = os.getenv("AWS_S3_MODEL_PREFIX", "ml-models").strip("/")
        download_dir_from_s3(local_path, f"{prefix}/{model_name}")

    # Importing the API and browsing /docs does not require loading Torch or AWS.
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(MODEL_TASKS[model_name], model=str(local_path), device=device)


def get_model(model_name):
    if model_name not in MODEL_TASKS:
        raise ValueError(f"Unknown model: {model_name}")
    # Keep simultaneous first requests from downloading/loading the same model.
    with _load_lock:
        return _load_model(model_name)
