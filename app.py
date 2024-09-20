#IMPORT STATEMENTS
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests

# Title of the app
st.title("Daisy/Dandelion Flower Classification")

# File uploader for the image
uploaded_image = st.file_uploader("Upload the Image", type=["png", "jpeg", "jpg"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image)

    # Load the trained model
    model_url = "https://raw.githubusercontent.com/Jesly-Joji/Flower-Image-Classification/main/cnn_image_classification_model.pt"
    response = requests.get(model_url)
    with open("model.pt", "wb") as f:
        f.write(response.content)

    # Load the model
    model = torch.load("model.pt")
    model.eval()  # Set the model to evaluation mode

    transform=transforms.Compose([
          transforms.Resize((28,28)),
          transforms.ToTensor()
          ])

    

    # Load the image and preprocess
    image = Image.open(uploaded_image)
    image_tensor=transform(image)
    image_tensor=image_tensor.unsqueeze(0)
   

    # Make predictions
    with torch.no_grad():
        pred=model(image_tensor)
       
    pred_index=pred.argmax(1)
    pred_class=classes[pred_index.item()]
  
    st.write(f"Predicted Class: {pred_class}")
