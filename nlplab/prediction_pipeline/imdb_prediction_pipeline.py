import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
#from loggin.logger import logging
import logging
import re
import contractions

class IMDBPredictionPipeline:
    def __init__(self, text: str, model: object):
        self.text = text
        self.model = model
        logging.info("Initialized IMDBPredictionPipeline.")

        self.word_index = imdb.get_word_index()
        self.reversed_word_index = {value: key for key, value in self.word_index.items()}

        self.MAX_WORDS = 10000

    
    def preprocess_text(self, text, maxlen=500):
        text = text.lower()

        # Expand contractions
        text = contractions.fix(text)

        # Remove punctuation (keep only letters and numbers)
        text = re.sub(r'[^a-z0-9\s]', '', text)
    
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return np.zeros((1, maxlen), dtype=np.int32)

        # ✅ Tokenize into words
        tokens = text.split()
        encoded_review = []
        for word in tokens:
            idx = self.word_index.get(word, 2) + 3
            if idx >= self.MAX_WORDS:
                idx = 2  # <UNK>
            encoded_review.append(idx)

        padded_review = sequence.pad_sequences(
            [encoded_review],
            maxlen=maxlen,
            padding='pre',
            truncating='post'
        )
        return np.array(padded_review, dtype=np.int32)

    def predict_tone(self, review):
        preprocessed_input = self.preprocess_text(review)
        prediction = self.model.predict(preprocessed_input, verbose=0)

        prob = float(prediction[0][0])

        if prob >= 0.65:
            sentiment = "Positive"
        elif prob <= 0.45:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment, prob
    
    def decode_review(self, encoded_review):
        decoded = []
        for i in encoded_review:
            word = self.reversed_word_index.get(i - 3, "<UNK>")  # fallback for unknown
            decoded.append(word)
        return " ".join(decoded)
    
    def decode_review_max(self, encoded_review):
        decoded = []
        for i in encoded_review:
            word = self.reversed_word_index.get(i - 3)  # no default
            if word:   # only add if not None
                decoded.append(word)
        return " ".join(decoded)
    


if __name__ == "__main__":
    # Dummy model placeholder (not used here, but required by class)
    class DummyModel: pass

    # Instantiate pipeline
    pipeline = IMDBPredictionPipeline(text="", model=DummyModel())

    # Test sentences
    test_sentences = [
        "How's the movie? I can't believe it's so good!",
        "Terrible film!!! Waste of time...",
        "It was okay, nothing special."
    ]

    for sentence in test_sentences:
        preprocessed = pipeline.preprocess_text(sentence)
        decoded_text = pipeline.decode_review_max(preprocessed[0])  # take first sequence
        #print(f"\nOriginal: {sentence}")
        #print(f"Preprocessed tokens: {preprocessed}")
        print(f"Decoded back: {decoded_text}")