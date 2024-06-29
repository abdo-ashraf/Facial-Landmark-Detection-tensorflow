## import required libraries
import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2

# Load a pre-trained object detection model
MODEL = load_model("D:/Eng/MLearning/Streamlit/Facial-Landmark-Detection/results/model.h5")
IMAGE_W = 512
IMAGE_H = 512

def load_and_preproces_image(image_file):
    
    ori_image = Image.open(image_file)
    ori_size = ori_image.size
    
    # Ensure the image is in RGB format
    processed_image = ori_image.convert("RGB")
    
    # Resize the image to the required input size
    processed_image = processed_image.resize((IMAGE_W, IMAGE_H))
    
    # Normalize the image array to have values between 0 and 1
    processed_image = np.array(processed_image) / 255.0
    
    return ori_image, processed_image, ori_size
    

def classify_image(processed_image):
    
    # Expand dimensions to match the input shape expected (Batch, IMAGE_W, IMAGE_H, 3)
    model_image = np.expand_dims(processed_image, axis=0)
    
    # Make predictions using the pre-trained MobileNetV2 model
    landmarks = MODEL.predict(model_image)
    
    for gg in landmarks.reshape(-1,2):
        gg = np.multiply(gg, [IMAGE_W, IMAGE_H]).astype(dtype=np.int16)
        mimage = cv2.circle(processed_image, (gg[0],gg[1]), 4, (0,1,0), -1)
    
    # Return Burned Image landmarks
    return mimage


# Streamlit app layout
st.title("Facial LandMark Detection")
st.write("Upload an image to detect its components")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    ori_image, proc_image, ori_size = load_and_preproces_image(uploaded_file)
    
    cols = st.columns(2)
    cols[0].image(ori_image, caption='Uploaded Image.', use_column_width=True)
    
    if cols[0].button('LandMarks Predict'):
        
        burned_image = classify_image(proc_image)
        burned_image = cv2.resize(burned_image, dsize=ori_size)
        
        cols[1].image(burned_image, caption='Burned Image.', use_column_width=True)
