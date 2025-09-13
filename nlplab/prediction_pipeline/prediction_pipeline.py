import re
import pandas as pd
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loggin.logger import logging
import zipfile

@st.cache_resource
def ensure_nltk():
    resources = ["wordnet", "omw-1.4", "stopwords", "words"]
    for r in resources:
        try:
            nltk.data.find(f"corpora/{r}")
        except (LookupError, zipfile.BadZipFile):
            logging.info(f"Downloading NLTK resource: {r}")
            nltk.download(r, quiet=True)
            logging.info(f"Downloaded NLTK resource: {r}")
    logging.info("All required NLTK resources are ready.")
    return True

ensure_nltk()

# ======================
# Preprocessing utils
# ======================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
english_vocab = set(words.words())

def validate_real_words(msg, threshold=0.5):
    """Validate that a message contains enough real English words."""
    tokens = [lemmatizer.lemmatize(w.lower()) for w in msg.split() if w.isalpha()]
    if not tokens:
        return False, "Message is empty or contains no valid words."
    
    real_words = [w for w in tokens if w in english_vocab]
    
    if len(real_words) / len(tokens) < threshold:
        return False, "Message contains too many unknown or random words."
    
    return True, ""

def validate_gibberish(msg):
    """Reject words with excessive consecutive consonants (likely gibberish)."""
    tokens = [w for w in msg.split() if w.isalpha()]
    for word in tokens:
        consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", word, re.I))
        if consonants / max(1, len(word)) > 0.7:
            return False, "Message contains gibberish."
    return True, ""

def validate_message(msg):
    """Combined validation for proper message."""
    if not msg.strip():
        return False, "Message is empty."
    
    # Real words check
    valid, err = validate_real_words(msg)
    if not valid:
        return False, err
    
    # Gibberish check
    valid, err = validate_gibberish(msg)
    if not valid:
        return False, err
    
    return True, ""

class PredictionPipeline:
    def __init__(self, text: str, model: object):
        self.text = text
        self.model = model
        logging.info("Initialized PredictionPipeline.")

    def preprocess_text(self):
        """Clean and lemmatize input text"""
        text = self.text.lower()
        logging.info(f"Raw Text: {text}")

        # Remove URLs
        text = re.sub(r"(?:https?|ftp|ssh)://\S+", "url", text)
        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)
        # Remove non-alphabetic characters
        text = re.sub("[^a-zA-Z]", " ", text)

        words = text.split()
        words = [
            lemmatizer.lemmatize(word, pos="v")
            for word in words
            if word not in stop_words
        ]

        text = " ".join(words)
        logging.info(f"Preprocessed/Cleaned Text: {text}")
        return text

    def predict_data(self):
        """Preprocess text and make prediction"""
        cleaned_text = self.preprocess_text()
        logging.info(f"Input to model: {cleaned_text}")

        try:
            prediction = self.model.predict([cleaned_text])
            prediction_probability = self.model.predict_proba([cleaned_text])

            logging.info(f"Prediction: {prediction}")
            logging.info(f"Prediction probabilities: {prediction_probability}")

            return [prediction, prediction_probability]

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return [None, None]



    
