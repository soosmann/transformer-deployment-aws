from email import header
import streamlit as st
import requests
from api import s3

API_URL = "http://localhost:8000/api/v1/"
headers = {"Content-Type": "application/json"}

st.title("Machine Learning Model Deployment at Server")

model = st.selectbox("Select Model", ["Sentiment Analysis", "Disaster Classifier", "Pose Classifier"])

if model == "Sentiment Analysis":
    text = st.text_area("Enter Your Review")
    user_id = st.text_input("Enter Your Email", "marcel@hofmann-owl.de")
    if st.button("Predict"):
        response = requests.post(API_URL + "sentiment_analysis", json={"texts": [text], "user_id": user_id}, headers=headers)
        if response.status_code == 200:
            output = response.json()
            st.write(output)
            st.info(f"Labels: {output['labels']}, Scores: {output['scores']}")
        else:
            st.error("Error in prediction")
elif model == "Disaster Classifier":
    text = st.text_area("Enter Your Tweet")
    user_id = st.text_input("Enter Your Email", "marcel@hofmann-owl.de")
    if st.button("Predict"):
        response = requests.post(API_URL + "disaster_classifier", json={"texts": [text], "user_id": user_id}, headers=headers)
        if response.status_code == 200:
            output = response.json()
            st.write(output)
            st.info(f"Labels: {output['labels']}, Scores: {output['scores']}")
        else:
            st.error("Error in prediction")
elif model == "Pose Classifier":
    select_file = st.radio("Select Image Source", ["Upload Image", "Use URL"])

    if select_file == "Use URL":
        url = st.text_input("Enter Image URL")
    else:
        image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        file_name = "data_download/temp.jpg"

        if image is not None:
            with open(file_name, "wb") as f:
                f.write(image.getvalue())
            url = s3.upload_image_to_s3(file_name)

    user_id = st.text_input("Enter Your Email", "marcel@hofmann-owl.de")

    if st.button("Predict"):
        # if select_file == "Use URL":
        response = requests.post(API_URL + "pose_classifier", json={"urls": [url], "user_id": user_id}, headers=headers)
        # else:
        #     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        #     if uploaded_file is not None:
        #         files = {"file": uploaded_file.getvalue()}
        #         response = requests.post(API_URL + "pose_classifier", files=files, data={"user_id": user_id}, headers=headers)

        if response.status_code == 200:
            output = response.json()
            st.write(output)
            st.info(f"Labels: {output['labels']}, Scores: {output['scores']}")
        else:
            st.error("Error in prediction")