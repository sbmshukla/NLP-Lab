import os
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import nltk

from nlplab.prediction_pipeline.prediction_pipeline import PredictionPipeline
from nlplab.loggin.logger import logging
from nlplab.exception.exception import handle_exception
from manager.bucketmanager import S3ModelManager
from dotenv import load_dotenv

load_dotenv()

# Deployment status
deployment_status = str(os.getenv("DEPLOYMENT_STATUS")).lower() == "true"

# Populate os.environ from Streamlit secrets if deployment
if deployment_status:
    for k, v in st.secrets.items():
        os.environ[k] = str(v)

# Initialize S3 manager
s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)

# Initialize UI log
if "logs" not in st.session_state:
    st.session_state.logs = []


def ui_log(message):
    """Log to both Python logger and Streamlit UI panel."""
    logging.info(message)
    st.session_state.logs.append(message)


# ======================
# Model Helpers
# ======================
def load_model(model_path):
    """Joblib deserializer with UI logging."""
    try:
        model = joblib.load(model_path)
        ui_log(f"Model deserialized from: {model_path}")
        return model
    except Exception as e:
        st.exception(e)
        ui_log(f"Failed to deserialize model: {model_path}")
        return None


def get_model(s3_key, model_dir="models"):
    """Load model based on deployment mode."""
    if deployment_status:
        ui_log("Deployment mode: pulling model directly from S3 (in-memory)")
        try:
            model = s3_manager.pull_model_in_memory(s3_key)
            if model:
                ui_log(f"Model loaded from S3: {s3_key}")
            else:
                ui_log(f"pull_model_in_memory returned None for {s3_key}")
            return model
        except Exception as e:
            st.exception(e)
            return None

    # Local/dev mode
    local_model_path = os.path.join(model_dir, os.path.basename(s3_key))
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(local_model_path):
        ui_log(f"Model found locally: {local_model_path}")
        return load_model(local_model_path)

    ui_log(f"Downloading model from S3: {s3_key} → {local_model_path}")
    try:
        s3_manager.pull_model(s3_key=s3_key, local_model_path=local_model_path)
        s3_manager.manage_local_models(model_dir=model_dir)
        if os.path.exists(local_model_path):
            ui_log(f"Download complete: {local_model_path}")
            return load_model(local_model_path)
        else:
            ui_log(f"File missing after download: {local_model_path}")
            return None
    except Exception as e:
        st.exception(e)
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
            text=f"{value}%",
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


# ======================
# Cache Wrapper
# ======================
@st.cache_resource
def _cached_model_loader(s3_key):
    return get_model(s3_key)


def cached_get_model(s3_key):
    model = _cached_model_loader(s3_key)
    ui_log(f"Using cached model for {s3_key}")
    return model


def clear_model_cache():
    _cached_model_loader.clear()
    st.session_state.logs = []
    ui_log("Cleared model cache and reset logs")


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
        "Select NLP Task", ["Spam Detection", "Twitter Sentiment Detection"]
    )
    if st.button("Clear Model Cache"):
        clear_model_cache()
    st.markdown("---")
    st.markdown(
        "Developed by [@sbmshukla](https://github.com/sbmshukla)",
        unsafe_allow_html=True,
    )

# Spam Detection Task
if task == "Spam Detection":
    st.subheader("Spam / Ham Detection")
    msg = st.text_area("Enter a message to classify:")
    predict_triggered = st.button("Predict Spam")

    if predict_triggered:
        if msg.strip():
            ui_log("Fetching model...")
            model = cached_get_model("models/spam_classifier.pkl")
            if model:
                ui_log("Model ready → running prediction")
                try:
                    pipeline = PredictionPipeline(msg, model)
                    prediction = pipeline.predict_data()
                except Exception as e:
                    st.exception(e)
                    ui_log(f"Prediction failed: {e}")
                    prediction = None

                if prediction:
                    result = prediction[0][0]
                    ham_percent = np.round(prediction[1][0][0] * 100, 5)
                    spam_percent = np.round(prediction[1][0][1] * 100, 5)

                    ui_log(
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

# Debug Log Panel
st.markdown("---")
st.text_area("Debug Log", "\n".join(st.session_state.logs), height=200)
