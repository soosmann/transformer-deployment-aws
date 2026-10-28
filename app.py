"""Earlier course step: run sentiment inference directly inside Streamlit."""

import os
from pathlib import Path

import streamlit as st

from api.models import get_model
from api.s3 import download_dir_from_s3

st.set_page_config(page_title="Sentiment Analysis · Direct Inference", page_icon="💬")
st.title("Sentiment Analysis")
st.caption("Learning example: S3 model download and inference inside Streamlit.")
st.info("For the full three-model API demo, run model_serve_streamlit.py instead.")

local_path = Path(os.getenv("MODEL_DIR", "ml-model-dir")) / "tinybert-sentiment-analysis"
prefix = os.getenv("AWS_S3_MODEL_PREFIX", "ml-models").strip("/")

if st.button("Download model from S3", disabled=local_path.is_dir()):
    try:
        with st.spinner("Downloading model..."):
            download_dir_from_s3(local_path, f"{prefix}/tinybert-sentiment-analysis")
        st.success("Model downloaded.")
    except Exception:
        st.error("Download failed. Check AWS_S3_BUCKET, your credentials, and the model prefix.")

text = st.text_area("Enter your review")
if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Enter a review first.")
    else:
        try:
            with st.spinner("Loading model and predicting..."):
                classifier = get_model("tinybert-sentiment-analysis")
                st.json(classifier(text, truncation=True))
        except Exception:
            st.error(
                "Model unavailable. Supply local trained artifacts or configure S3; see README.md."
            )
