"""Streamlit client for the FastAPI inference service."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from api import s3

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/").rstrip("/") + "/"

st.set_page_config(page_title="Transformer Model Lab", page_icon="🤖", layout="centered")
st.title("Transformer Model Lab")
st.caption("TinyBERT for text · Vision Transformer for images · FastAPI inference")
st.write("Explore three fine-tuned classifiers through a shared model-serving API.")

with st.sidebar:
    st.header("About this project")
    st.write(
        "A course-based project covering model training, S3 artifact storage, and containerized inference on AWS."
    )
    st.caption("Models load on first use. The first prediction can take longer.")
    st.markdown("[Course inspiration](https://www.udemy.com/course/nlp-with-bert-in-python/)")

model = st.selectbox(
    "Select model", ["Sentiment Analysis", "Disaster Classifier", "Pose Classifier"]
)
user_id = st.text_input(
    "Email (request metadata)",
    "demo@example.com",
    help="This field is validated by the API; it is not a login.",
)

text = ""
url = ""
uploaded_image = None
if model == "Sentiment Analysis":
    text = st.text_area("Review", placeholder="The story kept me interested from beginning to end.")
    endpoint = "sentiment_analysis"
elif model == "Disaster Classifier":
    text = st.text_area("Tweet", placeholder="Heavy rain has caused flooding in the city.")
    endpoint = "disaster_classifier"
else:
    endpoint = "pose_classifier"
    source = st.radio("Image source", ["Use URL", "Upload Image"], horizontal=True)
    if source == "Use URL":
        url = st.text_input("Image URL", placeholder="https://example.com/photo.jpg")
    else:
        uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        st.caption(
            "Uploads are stored in your configured S3 bucket; a temporary link is sent to the API."
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Selected image", use_column_width=True)

if st.button("Run prediction", type="primary"):
    if not user_id.strip():
        st.warning("Enter an email address for the request.")
        st.stop()
    if model != "Pose Classifier" and not text.strip():
        st.warning("Enter some text first.")
        st.stop()
    if model == "Pose Classifier" and not url.strip() and uploaded_image is None:
        st.warning("Provide an image URL or upload an image first.")
        st.stop()

    try:
        with st.spinner("Running inference (the first request also loads the model)..."):
            if uploaded_image is not None:
                with TemporaryDirectory() as temporary_dir:
                    file_name = Path(temporary_dir) / (
                        "upload" + Path(uploaded_image.name).suffix.lower()
                    )
                    file_name.write_bytes(uploaded_image.getvalue())
                    url = s3.upload_image_to_s3(file_name)
            payload = {"user_id": user_id}
            payload.update({"urls": [url]} if model == "Pose Classifier" else {"texts": [text]})
            response = requests.post(API_URL + endpoint, json=payload, timeout=(10, 300))
            response.raise_for_status()
            output = response.json()

        label, score, latency = st.columns(3)
        label.metric("Prediction", output["labels"][0])
        score.metric("Model confidence", f"{output['scores'][0]:.1%}")
        latency.metric("Inference time", f"{output['prediction_time']} ms")
        st.caption(
            "Confidence is a model score, not a guarantee of correctness. Inference time excludes model loading and network time."
        )
        with st.expander("API response"):
            st.json(output)
    except requests.HTTPError as exc:
        st.error(f"API returned HTTP {exc.response.status_code}.")
        with st.expander("Error details"):
            st.code(exc.response.text)
    except requests.RequestException:
        st.error(
            "Could not reach the API. Check that the server is running and API_URL is correct; the request may also have timed out."
        )
    except (BotoCoreError, ClientError, OSError, ValueError):
        st.error(
            "Image upload failed. Check AWS_S3_BUCKET, your AWS credentials, and S3 upload permissions."
        )

# Learning note: original experiment with a multipart image endpoint. The current
# JSON endpoint uses a URL (including S3 presigned URLs) instead. A multipart
# endpoint would need a matching server implementation.
# if select_file == "Use URL":
# else:
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(API_URL + "pose_classifier", files=files, data={"user_id": user_id}, headers=headers)
