from __future__ import division, print_function
import streamlit as st
from PIL import Image, ImageOps

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = 'resnet.hdf5'

model = load_model(MODEL_PATH)
def model_predict(img, MODEL_PATH):
    
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    
    image = img
    
    size = (150, 150)
    
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    
    return np.argmax(prediction, axis=1)

st.title("Cotton Plant Disease Prediction App ")

uploaded_file = st.file_uploader("Please upload an image....", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption ='Please upload an image....', use_column_width=True)
    st.write("")
    st.write("Predicting the output...")
    label = model_predict(image, 'resnet.hdf5')
    if label == 0:
        st.write("It is a diseased cotton leaf.")
    elif label == 1:
        st.write("It is a diseased cotton plant.")
    elif label == 2:
        st.write("It is a fresh cotton leaf.")
    else:
        st.write("It is a fresh cotton plant.")
