
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("AI_vs_Human_Image_Detector.keras")

st.title("AI vs Human-Generated Image Detector")

st.write("Upload an image to check whether it is AI-generated or real.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "AI Generated Image"
        score = confidence * 100
    else:
        result = "Real Image"
        score = (1-confidence) * 100

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {score:.2f}%")
