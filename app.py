import os
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import nltk

from nlplab.prediction_pipeline.prediction_pipeline import PredictionPipeline
from nlplab.prediction_pipeline.imdb_prediction_pipeline import IMDBPredictionPipeline
from nlplab.loggin.logger import logging
from nlplab.exception.exception import handle_exception
from manager.bucketmanager import S3ModelManager
from dotenv import load_dotenv
from tensorflow.keras.models import load_model as tf_load_model

load_dotenv()

# Deployment status
deployment_status = str(os.getenv("DEPLOYMENT_STATUS")).lower() == "true"

# Populate os.environ from Streamlit secrets if deployment
if deployment_status:
    for k, v in st.secrets.items():
        os.environ[k] = str(v)
        logging.info(f"Set env var from Streamlit secrets: {k}")

    # ======================
    # Streamlit UI - Internet & Model Switch Warning
    # ======================

    st.warning(
        "⚠️ Warning: Loading models from S3 will use internet data. "
        "Please avoid switching models repeatedly to save bandwidth."
    )


# Initialize S3 manager
s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)
logging.info("Initialized S3ModelManager")

# Initialize UI log
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_model_key" not in st.session_state:
    st.session_state.last_model_key = None


def ui_log(message):
    """Log to both Python logger and Streamlit UI panel."""
    logging.info(message)
    st.session_state.logs.append(message)


# ======================
# Model Helpers
# ======================
def load_model_old(model_path):
    """Joblib deserializer with UI logging."""
    try:
        model = joblib.load(model_path)
        ui_log(f"Model deserialized from: {model_path}")
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        st.exception(e)
        ui_log(f"Failed to deserialize model: {model_path}")
        logging.error(f"Failed to deserialize model: {model_path}, Exception: {e}")
        return None

def load_model(model_path):
    """Load model based on file type (.pkl → joblib, .h5 → Keras)."""
    try:
        if model_path.endswith(".pkl"):
            model = joblib.load(model_path)
            ui_log(f"Joblib model deserialized from: {model_path}")
        elif model_path.endswith(".h5"):
            model = tf_load_model(model_path)
            ui_log(f"Keras model loaded from: {model_path}")
        else:
            raise ValueError(f"Unsupported model type: {model_path}")

        logging.info(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        st.exception(e)
        ui_log(f"Failed to load model: {model_path}")
        logging.error(f"Failed to load model: {model_path}, Exception: {e}")
        return None

def get_model(s3_key, model_dir="models"):
    """Load model based on deployment mode."""
    if deployment_status:
        ui_log("Deployment mode: pulling model directly from S3 (in-memory)")
        logging.info(f"Fetching model from S3: {s3_key}")
        try:
            model = s3_manager.pull_model_in_memory(s3_key)
            if model:
                ui_log(f"Model loaded from S3: {s3_key}")
                logging.info(f"Model successfully loaded from S3: {s3_key}")
            else:
                ui_log(f"pull_model_in_memory returned None for {s3_key}")
                logging.warning(f"pull_model_in_memory returned None for {s3_key}")
            return model
        except Exception as e:
            st.exception(e)
            logging.error(f"Failed to load model from S3: {s3_key}, Exception: {e}")
            return None

    # Local/dev mode
    local_model_path = os.path.join(model_dir, os.path.basename(s3_key))
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(local_model_path):
        ui_log(f"Model found locally: {local_model_path}")
        logging.info(f"Model found locally: {local_model_path}")
        return load_model(local_model_path)

    ui_log(f"Downloading model from S3: {s3_key} → {local_model_path}")
    logging.info(f"Downloading model from S3: {s3_key}")
    try:
        s3_manager.pull_model(s3_key=s3_key, local_model_path=local_model_path)
        s3_manager.manage_local_models(model_dir=model_dir)
        if os.path.exists(local_model_path):
            ui_log(f"Download complete: {local_model_path}")
            logging.info(f"Download complete: {local_model_path}")
            return load_model(local_model_path)
        else:
            ui_log(f"File missing after download: {local_model_path}")
            logging.warning(f"File missing after download: {local_model_path}")
            return None
    except Exception as e:
        st.exception(e)
        logging.error(f"Failed to download model: {s3_key}, Exception: {e}")
        return None


def visulize_result(label, value, color="blue"):
    """Horizontal bar chart for probability visualization."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[value],
            y=[label],
            orientation="h",
            marker=dict(color=color),
            text=f"{value:.2f}%",
            textposition="inside",
            textfont=dict(color="white", size=14),
            insidetextanchor="start",
            hoverinfo="x+text",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=True),
        height=100,
        margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor="rgba(240,240,240,0.2)",
        plot_bgcolor="rgba(240,240,240,0.2)",
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)
    logging.info(f"Visualized result for {label}: {value:.2f}%")


# ======================
# Cache Wrapper
# ======================
@st.cache_resource
def _cached_model_loader(s3_key):
    return get_model(s3_key)


def cached_get_model(s3_key):
    """Load model and clear cache if a different model is requested."""
    last_key = st.session_state.get("last_model_key")
    if last_key != s3_key:
        _cached_model_loader.clear()
        ui_log(f"Cleared previous cached model: {last_key}")
        logging.info(f"Cleared previous cached model: {last_key}")
        st.session_state["last_model_key"] = s3_key
    model = _cached_model_loader(s3_key)
    ui_log(f"Using cached model for {s3_key}")
    logging.info(f"Using cached model for {s3_key}")
    return model


def clear_model_cache():
    """Clear model cache and reset logs."""
    _cached_model_loader.clear()
    st.session_state.logs = []
    st.session_state["last_model_key"] = None
    ui_log("Cleared model cache and reset logs")
    logging.info("Cleared model cache and reset logs")


# ======================
# Streamlit UI
# ======================
logging.info("App Started")
st.set_page_config(page_title="NLP LAB", layout="wide", page_icon="🧠")
st.markdown("<h2 style='text-align:center;'>NLP LAB</h2>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    task = st.selectbox(
        "Select NLP Task", ["Spam Detection", "Tweet Sentiment Detection", "Movie Review Classifier"]
    )
    if st.button("Clear Model Cache"):
        clear_model_cache()
    st.markdown("---")
    st.markdown(
        "Developed by [@sbmshukla](https://github.com/sbmshukla)",
        unsafe_allow_html=True,
    )

# ======================
# Spam Detection Task
# ======================
if task == "Spam Detection":
    st.subheader("Spam / Ham Detection")
    msg = st.text_area("Enter a message to classify:")
    predict_triggered = st.button("Predict Spam")

    if predict_triggered:
        if msg.strip():
            ui_log("Fetching model...")
            logging.info("Fetching spam detection model")
            model = cached_get_model("models/spam_classifier.pkl") if deployment_status else get_model("models/spam_classifier.pkl")
            if model:
                ui_log("Model ready → running prediction")
                logging.info("Running prediction for Spam Detection")
                try:
                    pipeline = PredictionPipeline(msg, model)
                    prediction = pipeline.predict_data()
                except Exception as e:
                    st.exception(e)
                    ui_log(f"Prediction failed: {e}")
                    logging.error(f"Prediction failed: {e}")
                    prediction = None

                if prediction:
                    result = prediction[0][0]
                    ham_percent = np.round(prediction[1][0][0] * 100, 2)
                    spam_percent = np.round(prediction[1][0][1] * 100, 2)

                    ui_log(
                        f"Prediction result: {'Spam' if result else 'Ham'}, Probabilities - Ham: {ham_percent}%, Spam: {spam_percent}%"
                    )
                    logging.info(
                        f"Prediction result: {'Spam' if result else 'Ham'}, Probabilities - Ham: {ham_percent}%, Spam: {spam_percent}%"
                    )

                    if result == 1:
                        st.error("Warning: Maybe It's Spam")
                    else:
                        st.success("Maybe It's Ham")

                    with st.expander("View Probability Details", expanded=False):
                        visulize_result(label="Spam", value=spam_percent, color="red")
                        visulize_result(label="Ham", value=ham_percent, color="green")
            else:
                st.error("Model could not be loaded.")
        else:
            st.warning("Please enter a message.")

# ======================
# Tweet Sentiment Detection Task
# ======================
elif task == "Tweet Sentiment Detection":
    st.subheader("Tweet Sentiment Detection")
    tweet = st.text_area("Enter a tweet to analyze:")
    predict_triggered = st.button("Predict Sentiment")

    if predict_triggered:
        if tweet.strip():
            ui_log("Fetching sentiment model...")
            logging.info("Fetching sentiment model")
            model = cached_get_model("models/sentiment_classifier.pkl") if deployment_status else get_model("models/sentiment_classifier.pkl")
            if model:
                ui_log("Sentiment model ready → running prediction")
                logging.info("Running prediction for Tweet Sentiment Detection")
                try:
                    pipeline = PredictionPipeline(tweet, model)
                    prediction = pipeline.predict_data()
                except Exception as e:
                    st.exception(e)
                    ui_log(f"Prediction failed: {e}")
                    logging.error(f"Prediction failed: {e}")
                    prediction = None

                if prediction:
                    result_code = prediction[0][0]
                    probabilities = prediction[1][0]

                    sentiment_map = {
                        2: "Positive",
                        1: "Neutral",
                        0: "Negative",
                        -1: "Irrelevant",
                    }
                    sentiment = sentiment_map.get(result_code, "Unknown")
                    st.success(f"Predicted Sentiment: {sentiment}")
                    ui_log(f"Prediction: {sentiment}, Probabilities: {probabilities}")
                    logging.info(
                        f"Prediction: {sentiment}, Probabilities: {probabilities}"
                    )

                    with st.expander("View Probability Details", expanded=False):
                        for code, color in zip(
                            [-1, 0, 1, 2], ["gray", "red", "blue", "green"]
                        ):
                            visulize_result(
                                label=sentiment_map[code],
                                value=(
                                    probabilities[code + 1] * 100
                                    if code >= 0
                                    else probabilities[0] * 100
                                ),
                                color=color,
                            )
            else:
                st.error("Sentiment model could not be loaded.")
        else:
            st.warning("Please enter a tweet.")



# ======================
# Movie Review Classifier Task Using RNN Deep Learning
# ======================
elif task == "Movie Review Classifier":
    st.subheader("Movie Review Classifier Using RNN-NLP Deep Learning")
    review = st.text_area("Enter a review to analyze:")
    predict_triggered = st.button("Classify Review")

    if predict_triggered:
        if review.strip():
            ui_log("Fetching RNN-NLP model...")
            logging.info("Fetching simple_rnn_imdb_v1.h5 model")
            
            model = cached_get_model("models/simple_rnn_imdb_v1.h5") if deployment_status else get_model("models/simple_rnn_imdb_v1.h5")
            
            if model:
                ui_log("RNN model ready → running prediction")
                logging.info("Running prediction for Movie Review Classifier")
                try:
                    # Use the review text instead of 'tweet'
                    pipeline :IMDBPredictionPipeline = IMDBPredictionPipeline(review, model)
                    prediction, prob = pipeline.predict_tone(review=review)
                    st.warning(f"{prediction} - {prob}")
                except Exception as e:
                    st.exception(e)
                    ui_log(f"Prediction failed: {e}")
                    logging.error(f"Prediction failed: {e}")

            else:
                st.error("RNN model could not be loaded.")
        else:
            st.warning("Please enter a review.")



# ======================
# Debug Log Panel
# ======================
st.markdown("---")
st.text_area("Debug Log", "\n".join(st.session_state.logs), height=200)
