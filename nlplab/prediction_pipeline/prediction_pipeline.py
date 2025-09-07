import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nlplab.loggin.logger import logging

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


class PredictionPipeline:
    def __init__(self, text: str, model: object):
        self.text = text
        self.model = model

    def preprocess_text(self):
        """Clean and lemmatize input text"""
        text = self.text.lower()
        logging.info(f"Raw Text: {text}")
        text = re.sub(r"(?:https?|ftp|ssh)://\S+", "url", text)  # remove URLs
        text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
        text = re.sub("[^a-zA-Z]", " ", text)  # remove non-alphabetic chars
        words = text.split()
        words = [
            lemmatizer.lemmatize(word, pos="v")
            for word in words
            if word not in stop_words
        ]
        text = " ".join(words)
        logging.info(f"Clean Text: {text}")
        return text

    def predict_data(self):
        """Preprocess text and make prediction"""
        cleaned_text = self.preprocess_text()
        # Wrap in list because most sklearn models expect 2D input
        return self.model.predict([cleaned_text])
