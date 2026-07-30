import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Tinea vs Candidiasis Classifier",
    page_icon="🩺",
    layout="centered"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Tinea_Candidiasis_MobileNetV3.keras")
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ==========================================================
# CLASS NAMES
# ==========================================================
CLASS_NAMES = ["Candidiasis", "Tinea"]

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================
def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(image)

    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==========================================================
# TITLE
# ==========================================================
st.title("🩺 Tinea vs Candidiasis Skin Disease Classifier")

st.write(
"""
This application uses a **Transfer Learning MobileNetV3 model**
to classify skin images into:

- **Tinea**
- **Candidiasis**

Upload a skin image below to obtain the prediction.
"""
)

# ==========================================================
# FILE UPLOADER
# ==========================================================
uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# PREDICTION
# ==========================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        if prediction >= 0.5:
            predicted_class = "Tinea"
            confidence = prediction * 100
        else:
            predicted_class = "Candidiasis"
            confidence = (1 - prediction) * 100

    st.success("Prediction Completed")

    st.subheader("Prediction Result")

    st.write(f"### Disease: **{predicted_class}**")

    st.write(f"### Confidence: **{confidence:.2f}%**")

    st.progress(float(confidence / 100))

    if predicted_class == "Tinea":

        st.info(
        """
### About Tinea

Tinea is a fungal infection affecting the skin, hair or nails.

Common symptoms include:

- Circular rash
- Redness
- Itching
- Scaling
"""
        )

    else:

        st.info(
        """
### About Candidiasis

Candidiasis is a fungal infection caused by Candida species.

Common symptoms include:

- Red rash
- Itching
- White patches
- Skin irritation
"""
        )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")

st.markdown(
"""
**GET324 – Cloud Computing and AI Model Deployment**

Mechanical & Aerospace Engineering

University of Uyo

Developed using:
- TensorFlow
- MobileNetV3
- Streamlit
"""
)
