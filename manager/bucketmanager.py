import os
import boto3
import time
import joblib
import io
from dotenv import load_dotenv

load_dotenv()


class S3ModelManager:
    def __init__(self, access_key, secret_key, region, bucket_name):
        self.deployment_status = os.getenv("DEPLOYMENT_STATUS")
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def pull_model_in_memory(self, s3_key):
        """Download model from S3 and load directly into memory"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            model_bytes = response["Body"].read()
            model = joblib.load(io.BytesIO(model_bytes))
            print(f"✅ Model loaded directly from S3 into memory: {s3_key}")
            return model
        except Exception as e:
            print(f"❌ Failed to load model from S3 into memory: {e}")
            return None

    def push_model(self, local_model_path, s3_key=None):
        """Upload model to S3"""
        if not s3_key:
            s3_key = os.path.basename(local_model_path)
        try:
            self.s3.upload_file(local_model_path, self.bucket_name, s3_key)
            print(f"✅ Uploaded '{local_model_path}' to S3 as '{s3_key}'")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

    def pull_model(self, s3_key, local_model_path=None):
        """Download model from S3"""
        if not local_model_path:
            local_model_path = os.path.join("models", os.path.basename(s3_key))
        try:
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            self.s3.download_file(self.bucket_name, s3_key, local_model_path)
            print(f"📦 Downloaded '{s3_key}' to '{local_model_path}'")
        except Exception as e:
            print(f"❌ Download failed: {e}")

    def manage_local_models(self, model_dir="models", max_models=2):
        """Ensure only the latest N models are kept locally"""
        if not os.path.exists(model_dir):
            return

        model_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f))
        ]

        if len(model_files) > max_models:
            # Sort by last modified time
            model_files.sort(key=lambda x: os.path.getmtime(x))
            files_to_delete = model_files[: len(model_files) - max_models]
            for file in files_to_delete:
                try:
                    os.remove(file)
                    print(f"🗑️ Deleted old model: {file}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {file}: {e}")
