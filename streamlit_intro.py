import select
import streamlit as st
import time
from PIL import Image

st.title("Machine Learning Model Deployment with Streamlit!")
st.header("This is a header")
st.subheader("This is a subheader")

st.text("This is a text element.")
input_text =st.text_input("Type something", "type here...")
st.text(input_text)
input_text = st.text_area("Type a longer text", "type here...")
st.markdown("This is a ___**markdown**___ element.")

button = st.button("Click me!")
if button:
    st.balloons()
    st.text("Button clicked!")
    st.toast("Button was clicked!")
    st.warning("This is a warning message!")
    st.error("This is an error message!")

st.image("https://www.streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=300)
img = Image.open("data/panda-1236875_640.jpg")
st.image(img, caption="Panda", use_column_width=True)
flag = st.checkbox("Check me!")
st.write(flag)
selection = st.radio("Choose your model", ["NLP", "Image", "Audio"])
st.write(f"You selected: {selection}")
st.selectbox("Select a number", [1, 2, 3, 4, 5])
st.multiselect("Select multiple options", ["Option 1", "Option 2", "Option 3"])

with st.spinner("Downloading..."):
    st.write("Downloading complete!")
    time.sleep(5)
st.slider("Select a range", 0, 100, (25, 75))