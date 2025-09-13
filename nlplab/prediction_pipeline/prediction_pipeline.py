import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from loggin.logger import logging
import zipfile

# Ensure NLTK resources are available
@st.cache_resource
def ensure_nltk():
    resources = ["wordnet", "omw-1.4", "stopwords"]
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

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


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



    
