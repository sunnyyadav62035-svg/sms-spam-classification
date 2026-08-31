import pickle
import string
from pathlib import Path

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -----------------------------
# Download NLTK resources
# -----------------------------
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# -----------------------------
# Initialize stemmer
# -----------------------------
ps = PorterStemmer()


# -----------------------------
# Text Transformation Function
# -----------------------------
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(ps.stem(i))

    # Return final text
    return " ".join(y)


# -----------------------------
# Load Model and Vectorizer
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error(
        "❌ model.pkl or vectorizer.pkl not found. "
        "Make sure both files are in the same folder as app.py."
    )
    st.stop()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)

st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message below to check whether it is **Spam** or **Not Spam**.")


# -----------------------------
# Input
# -----------------------------
input_sms = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict"):


        # Transform the SMS
        transformed_sms = transform_text(input_sms or " ")

        # Convert text into vector
        vector_input = tfidf.transform([transformed_sms])

        # Make prediction
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message appears to be a spam message.")

        else:
            st.success("✅ NOT SPAM")
            st.write("This message appears to be a legitimate message.")