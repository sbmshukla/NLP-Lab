🧠 NLP Lab
A modular, production-ready NLP application built with Streamlit, AWS S3, and custom prediction pipelines. This lab demonstrates scalable model deployment, dynamic loading, and interactive user feedback — all wrapped in a clean UI.

🚀 Features
Streamlit UI for real-time NLP tasks

Dynamic model loading from AWS S3 with local caching

Single-model memory mode for deployment environments

Spam/Ham classification with probability visualization

Modular architecture with reusable components

Robust logging and error handling

🗂️ Project Structure
NLP-Lab/
├── app.py                      # Streamlit frontend
├── manager/
│   └── bucketmanager.py        # S3 model manager
├── nlplab/
│   ├── prediction_pipeline/    # Custom prediction logic
│   ├── exception/              # Error handling
│   └── loggin/                 # Logging utilities
├── notebooks/                  # Exploratory notebooks
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup


🧪 Supported Tasks
📧 Spam Detection (Binary classification)

🛠️ Easily extendable to Sentiment Analysis, NER, etc.

🔐 Environment Setup
Create a .env file for local development:
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=your-region
AWS_S3_BUCKET_NAME=your-bucket-name
DEPLOYMENT_STATUS=False


For Streamlit ADD cRED iN sTREAMLIT PROJECT CLOUD SECRET

For deployment (e.g., Streamlit Cloud), set these as secrets and use:


DEPLOYMENT_STATUS=True


🧠 Model Management
Models are pulled from S3 only when needed

Local cache limited to 2 models (oldest auto-deleted)

In deployment mode, models are loaded directly into memory

📊 Visualization
Uses Plotly to display prediction confidence:

Horizontal bar charts for spam/ham probabilities

Expandable sections for detailed insights


🧰 Installation
bash
git clone https://github.com/sbmshukla/NLP-Lab.git
cd NLP-Lab
pip install -r requirements.txt
streamlit run app.py


👨‍💻 Author
Developed by @sbmshukla — passionate about clean ML pipelines, cloud-native deployment, and user-centric design.


📌 In Next Version
    - Add multi-task support (Sentiment, NER)

    - Integrate HuggingFace models

    - Add model versioning and metadata tracking

    - Dockerize for scalable deployment