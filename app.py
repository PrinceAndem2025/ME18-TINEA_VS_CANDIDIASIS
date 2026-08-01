"""
ME18: Tinea vs Candidiasis
--------------------------------------------
Streamlit web app. Loads the trained model and lets a user upload a
photo of a skin lesion to classify it as Tinea or Candidiasis.

Run locally with:
    streamlit run app.py
"""

import json

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = "Tinea_Candidiasis_MobileNetV3.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="Tinea vs Candidiasis Classifier", page_icon="🩺")


# ---------------------------------------------------------------------
# LOAD MODEL + CLASS NAMES (cached so it only loads once per session)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model_and_classes()

# Figure out which index is "Tinea" so display order is consistent
# regardless of alphabetical class ordering from Keras.
tinea_index = next(
    (i for i, name in enumerate(class_names) if "tinea" in name.lower()),
    0,
)
candidiasis_index = 1 - tinea_index


# ---------------------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------------------
def predict(model, img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    # The model already has preprocess_input built into its architecture
    raw_prob = model.predict(arr)[0][0]  # probability of class index 1 (Tinea)

    # --- CALIBRATION ---
    # The trained model is heavily imbalanced and biased towards Tinea 
    # (its recall for Candidiasis is only 36% while Tinea is 98%).
    # We apply a post-training log-odds shift to re-calibrate the probabilities.
    import math
    raw_prob_clipped = max(min(raw_prob, 0.9999), 0.0001)
    log_odds = math.log(raw_prob_clipped / (1 - raw_prob_clipped))
    
    # Shift the log-odds away from Tinea (class 1) to balance the bias
    # A shift of ~2.2 pushes an 89% Tinea prediction down to ~48% (making it Candidiasis)
    calibrated_log_odds = log_odds - 2.2
    prob = 1 / (1 + math.exp(-calibrated_log_odds))

    tinea_pct = (prob if tinea_index == 1 else 1 - prob) * 100
    candidiasis_pct = 100 - tinea_pct

    label = class_names[tinea_index] if tinea_pct >= 50 else class_names[candidiasis_index]

    return label, tinea_pct, candidiasis_pct


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🩺 Tinea vs Candidiasis Classifier")
st.write(
    "Upload a photo of a skin lesion, and this app will predict whether it "
    "shows signs consistent with **Tinea** (a fungal ringworm-type infection) "
    "or **Candidiasis** (a yeast/Candida infection)."
)

st.warning(
    "⚠️ **This tool is for educational purposes only and is NOT a medical "
    "diagnosis.** Skin conditions can look visually similar and require "
    "clinical examination and, often, lab testing (e.g. skin scraping/KOH "
    "prep) to distinguish accurately. Please consult a qualified healthcare "
    "provider or dermatologist for any real skin concern."
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    with st.spinner("Analysing image..."):
        label, tinea_pct, candidiasis_pct = predict(model, img)

    st.write(f"**Prediction:** {label}")

    st.progress(int(tinea_pct), text=f"Tinea: {tinea_pct:.1f}%")
    st.progress(int(candidiasis_pct), text=f"Candidiasis: {candidiasis_pct:.1f}%")
else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.caption(
    "Model: MobileNetV2 (transfer learning) · "
    "Dataset: Skin Disease Dataset (Kaggle, pacificrm) · "
    "GET 324 Mini-Project — Task ME18"
)
