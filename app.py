import streamlit as st

st.title("Daisy/ Dandellion Flower Classifcation")

uploaded_image = st.file_uploader("Upload the Image", type=["png","jpeg","jpg"])

st.image(uploaded_img)
