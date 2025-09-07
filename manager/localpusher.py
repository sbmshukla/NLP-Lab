from bucketmanager import S3ModelManager
import os
from dotenv import load_dotenv

load_dotenv()
# Get absolute path to project root (one level up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize manager
s3_manager = S3ModelManager(
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION"),
    bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
)


if __name__ == "__main__":
    s3_manager.push_model(
        local_model_path="models/spam_classifier.pkl",
        s3_key="models/spam_classifier.pkl",
    )
