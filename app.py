import streamlit as st
import joblib
import re
import nltk
import speech_recognition as sr
import time
from nltk.corpus import stopwords

# ===============================
# 🧩 Setup
# ===============================
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()
# Load ML model and vectorizer
model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Streamlit Config
st.set_page_config(page_title="AI Mental Health Detector", page_icon="🧠", layout="wide")

# ===============================
# 🎨 Custom CSS Styling
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #a1c4fd, #c2e9fb);
}
.main-title {
    text-align: center;
    color: #2E8B57;
    font-size: 45px;
    font-weight: bold;
    animation: fadeIn 1.5s;
}
.sub-title {
    text-align: center;
    color: #444;
    font-size: 18px;
}
@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}
.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 28px;
    margin-top: 15px;
}
.voice-btn {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 Header
# ===============================
st.markdown("<h1 class='main-title'>🧠 AI Mental Health Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Analyze your emotions through text or voice — powered by AI</p><br>", unsafe_allow_html=True)

# ===============================
# 🧹 Helper Functions
# ===============================
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([w for w in text.split() if w not in stopwords.words('english')])
    return text

# Speech recognition setup
recognizer = sr.Recognizer()

def record_and_transcribe():
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak for up to 10 seconds.")
        audio = recognizer.listen(source, phrase_time_limit=10)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand audio.")
            return ""
        except sr.RequestError:
            st.error("❌ Could not connect to Speech API.")
            return ""

# Initialize session state
if "voice_text" not in st.session_state:
    st.session_state["voice_text"] = ""

# ===============================
# ✍️ / 🎤 Input Section
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("✍️ Type Your Feelings")
    typed_text = st.text_area("Write here:", height=180, placeholder="How are you feeling today?")

with col2:
    st.subheader("🎤 Voice Mode")

    if st.button("🎙️ Record Voice (10s)"):
        with st.spinner("Converting speech to text..."):
            text = record_and_transcribe()
            if text:
                st.session_state["voice_text"] = text
                st.success("🗣️ You said: " + text)
                st.rerun() # Rerun to update the display immediately

if st.session_state["voice_text"]:
    st.info(f"🗣️ Last voice input: {st.session_state['voice_text']}")

# ===============================
# 🔍 Predict Emotion
# ===============================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💭 Analyze Emotion")

final_text = typed_text if typed_text.strip() != "" else st.session_state["voice_text"]

if st.button("🔮 Analyze Now"):
    if final_text.strip() == "":
        st.warning("⚠️ Please type or record some input first.")
    else:
        cleaned = clean_text(final_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Emotion color mapping
        emotion_colors = {
            "sadness": "#2196F3",
            "joy": "#4CAF50",
            "anger": "#F44336",
            "fear": "#9C27B0",
            "love": "#FF9800",
            "surprise": "#E91E63",
            "neutral": "#607D8B"
        }
        color = emotion_colors.get(prediction, "#333")

        st.markdown(f"""
        <div class='result-box' style='background-color:{color};'>
            Predicted Emotion: {prediction.upper()}
        </div>
        """, unsafe_allow_html=True)

        # Optional delay + emoji feedback
        time.sleep(0.3)
        emojis = {
            "joy": "😊",
            "sadness": "😢",
            "anger": "😠",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲",
            "neutral": "😐"
        }
        emoji = emojis.get(prediction, "🙂")
        st.markdown(f"<h2 style='text-align:center;'>Your mood: {emoji}</h2>", unsafe_allow_html=True)

st.markdown("<br><hr><p style='text-align:center; color:#555;'>Made with ❤️ using Streamlit, SpeechRecognition & ML</p>", unsafe_allow_html=True)
