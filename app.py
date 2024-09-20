import streamlit as st
import torch
import requests


st.title("Daisy/ Dandellion Flower Classifcation")

uploaded_image = st.file_uploader("Upload the Image", type=["png","jpeg","jpg"])

if uploaded_image:
   st.image(uploaded_image)


model_url="https://raw.githubusercontent.com/Jesly-Joji/Flower-Image-Classification/main/cnn_image_classification_model.pt"

response = requests.get(model_url)
with open("model.pt", "wb") as f:
    f.write(response.content)

torch.load("model.pt")
