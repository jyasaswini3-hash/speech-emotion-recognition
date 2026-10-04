import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# -----------------------------
# Load trained model components
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "rf_emotion_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    return model, scaler, label_encoder

rf_model, scaler, label_encoder = load_model()


# -----------------------------
# Feature extraction
# Must match training exactly
# -----------------------------
def extract_mfcc(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


# -----------------------------
# Prediction
# -----------------------------
def predict_emotion(file_path):
    features = extract_mfcc(file_path)
    features = features.reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = rf_model.predict(features_scaled)
    emotion = label_encoder.inverse_transform(prediction)[0]

    probabilities = None
    if hasattr(rf_model, "predict_proba"):
        probabilities = rf_model.predict_proba(features_scaled)[0]

    return emotion, probabilities


# -----------------------------
# User interface
# -----------------------------
st.title("🎤 Speech Emotion Recognition")
st.write(
    "Upload a speech recording and let the trained machine-learning model "
    "predict the emotion expressed in the voice."
)

st.info(
    "Supported emotions: Angry, Calm, Disgust, Fearful, Happy, "
    "Neutral, Sad, and Surprised."
)

st.subheader("📁 Upload an audio file")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "m4a", "webm"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔮 Predict Emotion", type="primary"):
        with st.spinner("Analyzing the speech..."):
            suffix = Path(uploaded_file.name).suffix or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            try:
                emotion, probabilities = predict_emotion(temp_path)

                st.success(f"🎤 Predicted Emotion: **{emotion.upper()}**")

                if probabilities is not None:
                    st.subheader("📊 Prediction Probabilities")

                    probability_data = {
                        emotion_name.upper(): float(probability)
                        for emotion_name, probability
                        in zip(label_encoder.classes_, probabilities)
                    }

                    st.bar_chart(probability_data)

            except Exception as e:
                st.error(
                    "The audio could not be processed. "
                    "Please try a clear speech recording."
                )
                st.exception(e)

            finally:
                Path(temp_path).unlink(missing_ok=True)


st.divider()


st.divider()

st.caption(
    "Machine Learning Model: Random Forest | "
    "Features: 40 MFCC mean coefficients | Dataset: RAVDESS"
)
