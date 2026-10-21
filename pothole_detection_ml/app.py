from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🚧",
    layout="centered",
)

st.title("🚧 Pothole Detection System")
st.write("Upload a road image to classify it as pothole or no pothole.")

MODEL_PATH = Path("models/pothole_model.keras")
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. First add images to both dataset folders "
        "and run `python train.py` from the project folder."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This image could not be opened. Please choose a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded road image", width="stretch")

    if st.button("🔍 Detect Pothole", type="primary"):
        with st.spinner("Analyzing image..."):
            resized = image.resize(IMG_SIZE)
            image_array = np.asarray(resized, dtype=np.float32)
            batch = np.expand_dims(image_array, axis=0)

            # Do NOT call preprocess_input here.
            # The matching training script puts MobileNetV2 preprocessing
            # inside the saved model. Preprocessing twice breaks predictions.
            output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

        if output.size != 1:
            st.error(
                "This app expects a binary classifier with one sigmoid output. "
                "The loaded model has an unexpected output shape."
            )
            st.stop()

        score = float(output[0])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            st.error(
                f"The model returned {score:.4f}, which is not a valid sigmoid "
                "probability. Check the model output layer."
            )
            st.stop()

        st.subheader("Prediction")
        st.metric("Model output (pothole score)", f"{score:.1%}")

        if score >= 0.5:
            st.error("🚧 POTHOLE DETECTED")
            st.caption(f"No-pothole score: {(1.0 - score):.1%}")
        else:
            st.success("✅ NO POTHOLE DETECTED")
            st.caption(f"No-pothole score: {(1.0 - score):.1%}")

        st.warning(
            "Student prototype only. The score is not a guarantee of correctness. "
            "The model must be trained and evaluated on correctly labeled, varied "
            "images before its predictions can be trusted."
        )
