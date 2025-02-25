import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("skin_disease_detect_V2.h5")

# Define class labels
class_labels = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Vascular lesion"
]

def preprocess_image(img):
    """
    Preprocess the uploaded image to match the input format required by the model.
    - Resizes the image to (64, 64)
    - Converts it to an array
    - Normalizes pixel values between 0 and 1
    - Expands dimensions to fit the model's expected input shape
    """
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_image(img):
    """
    Runs the preprocessed image through the model to obtain predictions.
    - Identifies the class with the highest probability
    - Returns the predicted class name and confidence score
    """
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence

st.title("Skin Disease Classifier")
st.write("""Simply upload an image of a skin lesion, and the model will analyze it to provide a predicted diagnosis.""")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image",  use_container_width=True)
    st.write("### Processing image... Please wait.")
    
    # Get prediction
    label, confidence = predict_image(img)
    
    # Display results~
    st.success(f"### Prediction: {label}")
    st.info(f"### Confidence: {confidence* 100:.2f}")
    
  