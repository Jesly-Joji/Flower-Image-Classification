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

    # Preprocess the uploaded image
    def preprocess_image(image):
        transform=transforms.Compose([
          transforms.Resize((28,28)),
          transforms.ToTensor()
          ])
        return transform(image).unsqueeze(0)  # Add batch dimension

    # Load the image and preprocess
    image = Image.open(uploaded_image)
    image_tensor = preprocess_image(image)

    # Make predictions
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    # Map the predicted class index to flower name
    class_names = ["Daisy", "Dandelion"]  # Adjust based on your model's output
    predicted_class = class_names[predicted.item()]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
