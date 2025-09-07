import os
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# ---- YOUR OWN MODULES ----
from nlplab.prediction_pipeline.prediction_pipeline import PredictionPipeline
from nlplab.loggin.logger import logging
from nlplab.exception.exception import handle_exception
from manager.bucketmanager import S3ModelManager

# ======================
# 1. Environment / Manager
# ======================
load_dotenv()

s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)

deployment_status = os.getenv("DEPLOYMENT_STATUS")  # "True" or "False"

# keep logs visible in UI
if "logs" not in st.session_state:
    st.session_state.logs = []


def ui_log(message):
    """Write to both Python logger and UI debug panel."""
    logging.info(message)
    st.session_state.logs.append(message)


ui_log(f"Deploy Status{deployment_status}")


# ======================
# 2. Model Helpers
# ======================
def load_model(model_path):
    """Joblib deserializer with UI error reporting."""
    try:
        model = joblib.load(model_path)
        ui_log(f"🔍 Model deserialized OK from {model_path}")
        return model
    except Exception as e:
        st.exception(e)
        return None


def get_model(s3_key, model_dir="models"):
    """
    - Deployment (DEPLOYMENT_STATUS == "True"): always pull in-memory
    - Dev mode: reuse local file or download
    """
    is_deploy = str(deployment_status).lower() == "true"

    if is_deploy:
        ui_log("🌐 Deployment mode → pulling directly from S3 (in-memory)")
        try:
            model = s3_manager.pull_model_in_memory(s3_key=s3_key)
            if model:
                ui_log("✅ Model loaded from S3 (in-memory)")
            else:
                ui_log("❌ pull_model_in_memory returned None")
            return model
        except Exception as e:
            st.exception(e)
            return None

    # ------- LOCAL / DEV MODE -------
    local_model_path = os.path.join(model_dir, os.path.basename(s3_key))
    if os.path.exists(local_model_path):
        ui_log(f"✅ Found locally: {local_model_path}")
        return load_model(local_model_path)

    ui_log(f"⬇️ Model not found locally → downloading `{s3_key}` → `{local_model_path}`")
    try:
        s3_manager.pull_model(s3_key=s3_key, local_model_path=local_model_path)
        s3_manager.manage_local_models(model_dir=model_dir)

        if os.path.exists(local_model_path):
            ui_log("📦 Download complete → deserializing")
            return load_model(local_model_path)
        else:
            ui_log("❌ File missing after download")
            return None
    except Exception as e:
        st.exception(e)
        return None


def visulize_result(label, value, color="blue"):
    """Horizontal bar for spam/ham probability."""
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
# 3. Cache Wrapper
# ======================
@st.cache_resource
def _cached_model_loader(s3_key):
    return get_model(s3_key)


def cached_get_model(s3_key):
    """Wrapper logs even when cache is hit."""
    model = _cached_model_loader(s3_key)
    ui_log(f"✅ Using cached model for `{s3_key}`")
    return model


def clear_model_cache():
    _cached_model_loader.clear()
    ui_log("🔄 Cleared model cache")


# ======================
# 4. Streamlit UI
# ======================
logging.info("App Started")

st.set_page_config(page_title="🧪 NLP LAB", layout="wide", page_icon="🧠")
st.markdown("<h2 style='text-align:center;'>💻 NLP LAB</h2>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    task = st.selectbox("🔍 Select NLP Task", ["Spam Detection"])
    if st.button("♻️ Clear Model Cache"):
        clear_model_cache()
    st.markdown("---")
    st.markdown(
        "Developed by [@sbmshukla](https://github.com/sbmshukla)",
        unsafe_allow_html=True,
    )

# ======================
# 5. Task: Spam Detection
# ======================
if task == "Spam Detection":
    st.subheader("📧 Spam / Ham Detection")

    msg = st.text_area("💬 Enter a message to classify:")
    predict_triggered = st.button("🚀 Predict Spam")

    if predict_triggered:
        if msg.strip():
            ui_log("🔎 Fetching model …")
            model = cached_get_model("models/spam_classifier.pkl")

            if model:
                ui_log("✅ Model ready → running prediction")
                try:
                    pipeline = PredictionPipeline(msg, model)
                    prediction = pipeline.predict_data()
                except Exception as e:
                    st.exception(e)
                    prediction = None

                if prediction:
                    result = prediction[0][0]  # predicted label
                    ham_percent = np.round(prediction[1][0][0] * 100, 5)
                    spam_percent = np.round(prediction[1][0][1] * 100, 5)

                    # ui_log(f"ham_percent: {ham_percent}, spam_percent: {spam_percent}")

                    if result == 1:
                        st.error("⚠️ Warning: Maybe It's Spam")
                    else:
                        st.success("✅ Maybe It's Ham")

                    with st.expander("🔍 View Probability Details", expanded=False):
                        visulize_result(label="Spam", value=spam_percent, color="red")
                        visulize_result(label="Ham", value=ham_percent, color="green")
            else:
                st.error("Model could not be loaded.")
        else:
            st.warning("⚠️ Please enter a message.")

# ======================
# 6. Debug Log Panel
# ======================
st.markdown("---")
st.text_area("📜 Debug Log", "\n".join(st.session_state.logs), height=200)
