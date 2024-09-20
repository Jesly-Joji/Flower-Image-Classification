import streamlit as st

st.title("Daisy/ Dandellion Flower Classifcation")

uploaded_image = st.file_uploader("Upload the Image", type=["png","jpeg","jpg"])

if uploaded_image:
   st.image(uploaded_image)
