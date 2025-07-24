from transformers import pipeline
import streamlit as st
import os
import boto3
import torch

bucket_name = "mlops-mhofmann"
local_path = 'data_download/tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

s3 = boto3.client('s3')
def download_dir_from_s3(bucket_name, local_path, s3_prefix):
    """Download a directory from an S3 bucket"""
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for obj in result["Contents"]:
                s3_key = obj["Key"]
                print(s3_key)
                local_file_path = os.path.join(local_path, s3_key.replace(s3_prefix, ""))
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file_path)
                print(f"Downloaded {obj['Key']} to {local_file_path}.")

st.title("Machine Learning Model Deployment at Server")           
button = st.button("Download Model")
if button:
    with st.spinner("Downloading model..."):
        download_dir_from_s3(bucket_name, local_path, s3_prefix)

text = st.text_area("Enter Your Review")
predict = st.button("Predict")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("sentiment-analysis", model="data_download/tinybert-sentiment-analysis", device=device)
if predict:
    with st.spinner("Predicting..."):
        output = classifier(text)
        st.write(output)
        st.info(output)