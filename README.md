# 🎤 Speech Emotion Recognition

A machine-learning web application that recognizes emotions from speech audio.

## Project Overview

This project uses the RAVDESS speech emotion dataset and a Random Forest classifier to classify speech into eight emotions:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Machine Learning Pipeline

1. Load speech audio at 22,050 Hz
2. Extract 40 MFCC features
3. Calculate the mean of each MFCC across time
4. Scale the features using StandardScaler
5. Predict using a trained Random Forest classifier
6. Convert the predicted label back to the emotion name

The trained model achieved approximately 65.97% accuracy on the held-out test set.

## Web Application

The application is built with Streamlit and supports:

- Audio file upload
- Browser microphone recording
- Emotion prediction
- Prediction probability visualization

## Project Files

- `app.py` — Streamlit web application
- `rf_emotion_model.pkl` — trained Random Forest model
- `scaler.pkl` — trained StandardScaler
- `label_encoder.pkl` — trained emotion label encoder
- `requirements.txt` — Python dependencies

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).

## Note

Speech emotion recognition is probabilistic. Predictions can vary depending on the speaker, recording quality, background noise, and similarity between the speaker's voice and the training data.
