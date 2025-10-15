import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import numpy as np
from scipy.sparse import hstack

# ===============================
# Setup
# ===============================
st.set_page_config(page_title="Mental Health Emotion Detector", page_icon="🧠", layout="wide")
nltk.download('stopwords', quiet=True)

# Load Model and Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load("model/emotion_model_final.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    emotion_labels = {
        "anger": "😡",
        "fear": "😨",
        "joy": "😄",
        "love": "❤️",
        "sadness": "😢",
        "surprise": "😮"
    }
    return model, vectorizer, emotion_labels

model, vectorizer, emotion_labels = load_assets()
vader = SentimentIntensityAnalyzer()

# Stopwords (keep negations)
stop_words = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                  'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                  "hasn't", 'haven', "haven't", 'isn', "isn't", 'wasn', "wasn't",
                  'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
final_stop_words = stop_words - negation_words

# ===============================
# Helper Functions
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join([w for w in text.split() if w not in final_stop_words])

def predict_emotion(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    
    # Add VADER compound score
    v_score = vader.polarity_scores(text)['compound']
    v_score_array = np.array([[v_score]])
    
    X_final = hstack([vectorized, v_score_array])
    
    model_pred = model.predict(X_final)[0]

    # Rule-based VADER override
    positive_labels = {"joy", "love", "surprise"}
    negative_labels = {"sadness", "anger", "fear"}
    if v_score <= -0.35 and model_pred in positive_labels:
        final_pred = "sadness"
    elif v_score >= 0.35 and model_pred in negative_labels:
        final_pred = "joy"
    else:
        final_pred = model_pred

    return final_pred, vader.polarity_scores(text)

# ===============================
# Voice Recording
# ===============================
recognizer = sr.Recognizer()
mic = sr.Microphone()

if "voice_text" not in st.session_state:
    st.session_state["voice_text"] = ""

def record_voice():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        st.info("🎙️ Recording... Speak now.")
        audio = recognizer.listen(source)  # no time limit
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
            return ""
        except sr.RequestError:
            st.error("Speech API unavailable.")
            return ""

# ===============================
# UI
# ===============================
st.markdown("""
<h1 style='text-align:center;color:#2E8B57;'>🧠 Mental Health Emotion Detector</h1>
<p style='text-align:center;color:#555;'>Analyze emotions from text or voice with AI</p>
""", unsafe_allow_html=True)

# Determine the initial value for the text area.
# If there's text from a voice recording, use that.
initial_text = st.session_state.get("text_area", "")
if "voice_transcription" in st.session_state:
    initial_text = st.session_state.pop("voice_transcription")

col1, col2 = st.columns([2,1])

# Text Input
with col1:
    st.subheader("✍️ Type Your Text")
    text_input = st.text_area("Enter your message here:", height=180)
    text_input = st.text_area("Enter your message here:", value=initial_text, height=180, key="text_area")

# Voice Input
with col2:
    st.subheader("🎤 Voice Input")
    if st.button("🎙️ Record Voice"):
        st.session_state["voice_text"] = record_voice()
        if st.session_state["voice_text"]:
            st.success(f"🗣️ You said: {st.session_state['voice_text']}")
        transcribed_text = record_voice()
        if transcribed_text:
            st.session_state["voice_transcription"] = transcribed_text
            st.success(f"🗣️ You said: \"{transcribed_text}\"")
            st.rerun()

# Final Text (voice or typed)
final_text = text_input.strip() or st.session_state.get("voice_text", "").strip()

# Prediction
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💭 Predict Emotion")

if st.button("🔮 Analyze Emotion"):
    if not final_text:
    if not text_input.strip():
        st.warning("⚠️ Please enter text or record your voice first.")
    else:
        pred, vader_scores = predict_emotion(final_text)
        pred, vader_scores = predict_emotion(text_input)
        emoji = emotion_labels.get(pred, "❓")
        st.markdown(f"<h2 style='text-align:center;color:#fff;background-color:#4CAF50;padding:15px;border-radius:10px;'>Predicted Emotion: {pred.upper()} {emoji}</h2>", unsafe_allow_html=True)

        with st.expander("📊 Detailed VADER Scores"):
            st.write(vader_scores)
            st.markdown(f"""
            - **Positive:** `{vader_scores['pos']:.2f}`
            - **Neutral:** `{vader_scores['neu']:.2f}`
            - **Negative:** `{vader_scores['neg']:.2f}`
            - **Compound Score:** `{vader_scores['compound']:.2f}`
            """)

st.markdown("<hr><p style='text-align:center;color:#555;'>Made with ❤️ using Streamlit + SpeechRecognition + ML</p>", unsafe_allow_html=True)
