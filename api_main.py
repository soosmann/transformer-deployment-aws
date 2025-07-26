from api.data_model import NLPDataInput, ImageDataInput, NLPDataOutput, ImageDataOutput
from api import s3
from fastapi import FastAPI
import uvicorn
import os
import torch
from transformers import pipeline, AutoImageProcessor
import time

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################ Model Download Start ################
force_download = False

model_name = "tinybert-sentiment-analysis/"
local_path = os.path.join("ml-model-dir", model_name)
s3_path = os.path.join("ml-models", model_name)
if not os.path.isdir(local_path) or force_download:
    print("sdakfjsk")
    s3.download_dir_from_s3(local_path, s3_path)
sentiment_model = pipeline("sentiment-analysis", model=local_path, device=device)

model_name = "tinybert-disaster-tweets/"
local_path = os.path.join("ml-model-dir", model_name)
s3_path = os.path.join("ml-models", model_name)
if not os.path.isdir(local_path) or force_download:
    s3.download_dir_from_s3(local_path, s3_path)
disaster_model = pipeline("text-classification", model=local_path, device=device)


model_name = "vit-human-pose-classification/"
local_path = os.path.join("ml-model-dir", model_name)
s3_path = os.path.join("ml-models", model_name)
vit_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(vit_ckpt, use_fast=True)
if not os.path.isdir(local_path) or force_download:
    s3.download_dir_from_s3(local_path, s3_path)
pose_model = pipeline("image-classification", model=local_path, image_processor=image_processor, device=device)
################ Model Download Start ################



@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput) -> NLPDataOutput:
    t0 = time.time()
    output = sentiment_model(data.texts)
    dt = int((time.time() - t0) * 1000)  # in milliseconds

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    return NLPDataOutput(
        pred_model="tinybert-sentiment-analysis",
        texts=data.texts,
        labels=labels,
        scores=scores,
        prediction_time=dt
    )

@app.post("/api/v1/disaster_classifier")
def disaster_classifier(data: NLPDataInput) -> NLPDataOutput:
    t0 = time.time()
    output = disaster_model(data.texts)
    dt = int((time.time() - t0) * 1000)

    labels = [x["label"] for x in output]
    scores = [x["score"] for x in output]

    return NLPDataOutput(
        pred_model="tinybert-disaster-tweets",
        texts=data.texts,
        labels=labels,
        scores=scores,
        prediction_time=dt
    )

@app.post("/api/v1/pose_classifier")
def pose_classifier(data: ImageDataInput) -> ImageDataOutput:
    t0 = time.time()
    output = pose_model(data.urls) 
    dt = int((time.time() - t0) * 1000)

    labels = [x[0]["label"] for x in output]
    scores = [x[0]["score"] for x in output]

    return ImageDataOutput(
        pred_model="vit-human-pose-classification",
        urls=data.urls,
        labels=labels,
        scores=scores,
        prediction_time=dt
    )

# when errors, use: uvicorn api_main:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run(app=app, port=8000, reload=True)