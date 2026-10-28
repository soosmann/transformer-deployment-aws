"""FastAPI entry point for the course's three trained classifiers."""

import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException

from api.data_model import ImageDataInput, ImageDataOutput, NLPDataInput, NLPDataOutput
from api.models import get_model

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Transformer Model Deployment API",
    description="Course-based TinyBERT and ViT inference with local or S3 model artifacts.",
    version="1.0.0",
)

# ################ Model Download Start ################
# Learning note: model downloads originally ran at import time. They now live in
# api/models.py and run on the first prediction, so /docs works without weights.
# Set AWS_S3_BUCKET for S3 downloads, or put trained artifacts in MODEL_DIR.
# ################ Model Download End ################


def load_classifier(model_name):
    try:
        return get_model(model_name)
    except Exception as exc:
        logger.exception("Could not load model %s", model_name)
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Provide trained artifacts in MODEL_DIR or configure AWS_S3_BUCKET. See README.md.",
        ) from exc


@app.get("/health", tags=["Service"])
def health():
    """Liveness only: model loading is deferred until a prediction is requested."""
    return {"status": "ok", "model_loading": "on_first_request"}


def classify_text(data, model_name):
    classifier = load_classifier(model_name)
    t0 = perf_counter()
    output = classifier(data.texts, truncation=True)
    dt = int((perf_counter() - t0) * 1000)  # in milliseconds
    return NLPDataOutput(
        pred_model=model_name,
        texts=data.texts,
        labels=[item["label"] for item in output],
        scores=[item["score"] for item in output],
        prediction_time=dt,
    )


@app.post("/api/v1/sentiment_analysis", tags=["Inference"])
def sentiment_analysis(data: NLPDataInput) -> NLPDataOutput:
    return classify_text(data, "tinybert-sentiment-analysis")


@app.post("/api/v1/disaster_classifier", tags=["Inference"])
def disaster_classifier(data: NLPDataInput) -> NLPDataOutput:
    return classify_text(data, "tinybert-disaster-tweets")


@app.post("/api/v1/pose_classifier", tags=["Inference"])
def pose_classifier(data: ImageDataInput) -> ImageDataOutput:
    classifier = load_classifier("vit-human-pose-classification")
    urls = [str(url) for url in data.urls]
    t0 = perf_counter()
    try:
        output = classifier(urls, timeout=10)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Could not read one of the supplied images."
        ) from exc
    dt = int((perf_counter() - t0) * 1000)
    return ImageDataOutput(
        pred_model="vit-human-pose-classification",
        urls=urls,
        labels=[item[0]["label"] for item in output],
        scores=[item[0]["score"] for item in output],
        prediction_time=dt,
    )


# when errors, use: uvicorn api_main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_main:app", host="127.0.0.1", port=8000, reload=True)
