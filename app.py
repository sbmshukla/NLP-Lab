import streamlit as st
import joblib
import numpy as np
from nlplab.prediction_pipeline.prediction_pipeline import PredictionPipeline
from nlplab.loggin.logger import logging
from nlplab.exception.exception import handle_exception
import plotly.graph_objects as go
from manager.bucketmanager import S3ModelManager
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize manager
s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)

deployment_status = os.getenv("DEPLOYMENT_STATUS")


def get_model(s3_key, model_dir="models"):
    local_model_path = os.path.join(model_dir, os.path.basename(s3_key))

    with st.spinner("🔄 Loading model..."):
        if not os.path.exists(local_model_path):
            st.info(f"⬇️ Model not found locally. Downloading: `{s3_key}`")
            if deployment_status == "True":
                model = s3_manager.pull_model_in_memory(s3_key=s3_key)
            else:
                s3_manager.pull_model(s3_key=s3_key, local_model_path=local_model_path)
                s3_manager.manage_local_models(model_dir=model_dir)
        else:
            # st.success(f"✅ Model already exists: `{local_model_path}`")
            pass

        model = load_model(local_model_path)

    return model


# Log app start
logging.info("App Started")


# Page config
st.set_page_config(page_title="🧪 NLP LAB", layout="wide", page_icon="🧠")

# Header
st.markdown("<h2 style='text-align:center;'>💻 NLP LAB</h2>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    task = st.selectbox("🔍 Select NLP Task", ["Spam Detection"])
    st.markdown("---")
    st.markdown(
        "Developed by [@sbmshukla](https://github.com/sbmshukla)",
        unsafe_allow_html=True,
    )


# Helper function to load model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        handle_exception(e)
    return None


def visulize_result(label, value, color="blue"):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[value],
            y=[label],
            orientation="h",
            marker=dict(color=color),
            text=f"{value}%",
            textposition="inside",
            textfont=dict(color="white", size=14),  # High contrast text
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


# -------------------------
# Task: Spam Detection
# -------------------------
if task == "Spam Detection":
    st.subheader("📧 Spam / Ham Detection")

    msg = st.text_area("💬 Enter a message to classify:")
    predict_triggered = st.button("🚀 Predict Spam")

    if predict_triggered:
        if msg.strip():
            # model = load_model("models/spam_classifier.pkl") #local approach
            model = get_model("models/spam_classifier.pkl")
            if model:
                pipeline = PredictionPipeline(msg, model)
                prediction = pipeline.predict_data()

                result = prediction[0][0]  # predicted label
                ham_percent = np.round(prediction[1][0][0] * 100, 5)
                spam_percent = np.round(prediction[1][0][1] * 100, 5)

                logging.info(
                    f"ham_percent: {ham_percent}, spam_percent: {spam_percent}"
                )
                if result == 1:
                    st.error("⚠️ Warning: Maybe It's Spam")
                else:
                    st.success("✅ Maybe It's Ham")

                with st.expander("🔍 View Probability Details", expanded=False):

                    visulize_result(label="Spam", value=spam_percent, color="red")
                    visulize_result(label="Ham", value=ham_percent, color="green")

        else:
            st.warning("⚠️ Please enter a message.")
