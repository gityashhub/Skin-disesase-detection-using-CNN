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

# Define advice dictionary
advice_dict = {
    "Actinic keratosis": [
        ("☀️ Prevention", "Minimize sun exposure, especially during peak hours."),
        ("👒 Protection", "Wear sun-protective clothing and wide-brimmed hats."),
        ("🧴 Sun Safety", "Use sunscreen on exposed areas of the body and face.")
    ],
    "Basal cell carcinoma": [
        ("⚕️ Treatment", "Surgical excision is usually curative."),
        ("🔍 Follow-up Care", "Long-term monitoring is necessary to detect recurrences early.")
    ],
    "Benign keratosis": [
        ("🔎 Monitoring", "Regular skin exams for early detection of changes."),
        ("☀️ Sun Protection", "Use sunscreen and avoid excessive sun exposure to prevent new lesions.")
    ],
    "Dermatofibroma": [
        ("🚫 Avoid Scratching", "Scratching can cause infections."),
        ("🫧 Hygiene", "Keep the affected area clean to prevent complications.")
    ],
    "Melanocytic nevus": [
        ("☀️ Sun Safety", "Protect your skin from UV rays."),
        ("🚫 Avoid Tanning Beds", "Minimize artificial UV exposure.")
    ],
    "Melanoma": [
        ("🧴 Sun Protection", "Wear protective clothing and use sunscreen with a high SPF."),
        ("🔎 Early Detection", "Monitor moles and spots for any changes.")
    ],
    "Squamous cell carcinoma": [
        ("☀️ Limit Sun Exposure", "Avoid the sun during peak UV index hours."),
        ("🧴 Sunscreen Use", "Apply SPF 30+ sunscreen, reapplying every 2 hours or after swimming/sweating."),
        ("📅 Regular Checkups", "Schedule annual dermatology visits and get immediate assessments for new or changing spots.")
    ],
    "Vascular lesion": [
        ("⚕️ Medical Advice", "Consult a dermatologist for assessment and potential treatment options."),
        ("🩹 Wound Care", "Keep the area clean and avoid trauma to prevent bleeding or infection.")
    ]
}

def preprocess_image(img):
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array


def predict_image(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence


def display_advice(disease):
    if disease in advice_dict:
        st.subheader("🩺 Advice for " + disease)
        st.markdown("---")
        for point, description in advice_dict[disease]:
            st.info(f"**{point}:** {description}")
        st.markdown("---")


st.title("Skin Disease Classifier")
st.write("""Upload an image of a skin lesion, and the model will analyze it to provide a predicted diagnosis, confidence score, and medical advice.""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("### Processing image... Please wait.")

    # Get prediction
    label, confidence = predict_image(img)

    # Display results
    st.success(f"### 🩻 Prediction: {label}")
    st.info(f"### 📊 Confidence: {confidence * 100:.2f}%")

    # Show advice
    display_advice(label)

  
