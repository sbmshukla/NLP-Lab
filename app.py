import streamlit as st
import joblib
import os
from prediction_pipeline.prediction_pipeline import PredictionPipeline

# Page config
st.set_page_config(page_title="🧪 NLP LAB", layout="wide", page_icon="🧠")

# Header
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>💻 NLP LAB</h1>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Sidebar
with st.sidebar:

    task = st.selectbox(
        "🔍 Select NLP Task",
        [
            "Spam Detection",
            # "Sentiment Analysis",
            # "Text Summarization",
            # "Toxic Comment Detection",
        ],
    )
    st.markdown("---")
    st.markdown(
        "Developed by [@sbmshukla](https://github.com/sbmshukla)",
        unsafe_allow_html=True,
    )


# Helper function placeholder
def load_model(model_path):
    return joblib.load(model_path)


# Task Containers
if task == "Spam Detection":
    with st.container():
        st.subheader("📧 Spam / Ham Detection")
        msg = st.text_area("💬 Enter a message to classify:")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 Predict Spam"):
                if msg.strip():

                    model = load_model("models/spam_classifier.pkl")
                    # prediction = model.predict([msg])
                    pipeline = PredictionPipeline(msg, model)
                    prediction = pipeline.predict_data()[0]
                    st.success(f"Prediction: {'Spam' if prediction==0 else 'Ham'}")
                else:
                    st.warning("⚠️ Please enter a message.")

elif task == "Sentiment Analysis":
    with st.container():
        st.subheader("😊 Sentiment Analysis")
        text = st.text_area("📝 Enter text to analyze sentiment:")

        if st.button("🔍 Predict Sentiment"):
            if text.strip():
                st.success("Prediction placeholder: Positive/Negative/Neutral")
            else:
                st.warning("⚠️ Please enter text.")
