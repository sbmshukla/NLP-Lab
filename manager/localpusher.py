from bucketmanager import S3ModelManager
import os
from dotenv import load_dotenv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nlplab.loggin.logger import logging

# Load environment variables
load_dotenv()

# Get absolute path to project root (one level up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize S3 manager
s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)


if __name__ == "__main__":
    local_model_path = os.path.join(PROJECT_ROOT, "models", "simple_rnn_imdb_v1.h5")
    s3_key = "models/simple_rnn_imdb_v1.h5"

    logging.info(f"Starting upload of model: {local_model_path} to S3 as {s3_key}")
    try:
        s3_manager.push_model(local_model_path=local_model_path, s3_key=s3_key)
        logging.info(f"Model upload completed successfully: {s3_key}")
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")
