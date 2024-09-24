import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('cnn_tumor.h5')

def make_prediction(img, model):
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1)
    res = model.predict(input_img)
    if res > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"

# Streamlit app
st.title("Tumor Detection from MRI Images")
st.write("Upload a JPG image to detect if it has a tumor.")

uploaded_file = st.file_uploader("Choose a JPG image...", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    
    # Display the uploaded image
    st.image(img, channels="RGB")
    
    # Make prediction
    result = make_prediction(img, model)
    st.write(result)



    