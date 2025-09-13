import os
import boto3
import joblib
import io
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nlplab.loggin.logger import logging
from tensorflow.keras.models import load_model
import h5py
from nlplab.exception.exception import CustomException
import tempfile
class S3ModelManager:
    def __init__(self, access_key, secret_key, region, bucket_name):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def pull_model_in_memory_old(self, s3_key):
        """Download model from S3 and load directly into memory"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            model_bytes = response["Body"].read()
            model = joblib.load(io.BytesIO(model_bytes))
            logging.info(f"Model loaded directly from S3 into memory: {s3_key}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model from S3 into memory: {e}")
            return None

    def pull_model_in_memory_semi_old(self, s3_key):
        """Download model from S3 and load directly into memory"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            model_bytes = response["Body"].read()
            ext = os.path.splitext(s3_key)[-1].lower()

            if ext in [".h5", ".keras"]:  # TensorFlow/Keras model
                with io.BytesIO(model_bytes) as f:
                    model = load_model(f)  # <- pass BytesIO directly
                logging.info(f"Keras model loaded from S3 into memory: {s3_key}")
            # elif ext in [".h5", ".keras"]:  # TensorFlow/Keras model
                
            #     with io.BytesIO(model_bytes) as f:
            #         with h5py.File(f, "r") as h5file:
            #             model = load_model(h5file)
            #     logging.info(f"Keras model loaded from S3 into memory: {s3_key}")

            elif ext in [".pkl", ".joblib"]:  # sklearn/joblib models
                model = joblib.load(io.BytesIO(model_bytes))
                logging.info(f"Joblib model loaded from S3 into memory: {s3_key}")

            else:
                logging.error(f"Unsupported model format: {ext}")
                return None

            return model

        except Exception as e:
            logging.error(f"Failed to load model from S3 into memory: {e}")
            raise CustomException(e, sys)
            #return None
    def pull_model_in_memory(self, s3_key):
        """Download model from S3 and load directly into memory"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            model_bytes = response["Body"].read()
            ext = os.path.splitext(s3_key)[-1].lower()

            if ext in [".h5", ".keras"]:  # TensorFlow/Keras model
                # Write to a temporary file
                with tempfile.NamedTemporaryFile(suffix=ext) as tmp_file:
                    tmp_file.write(model_bytes)
                    tmp_file.flush()
                    model = load_model(tmp_file.name)  # load from temp path
                logging.info(f"Keras model loaded from S3 into memory: {s3_key}")

            elif ext in [".pkl", ".joblib"]:  # sklearn/joblib models
                model = joblib.load(io.BytesIO(model_bytes))
                logging.info(f"Joblib model loaded from S3 into memory: {s3_key}")

            else:
                logging.error(f"Unsupported model format: {ext}")
                return None

            return model

        except Exception as e:
            logging.error(f"Failed to pull model in memory: {e}")
            raise CustomException(e, sys)
            #return None

    def push_model(self, local_model_path, s3_key=None):
        """Upload model to S3"""
        if not s3_key:
            s3_key = os.path.basename(local_model_path)
        try:
            self.s3.upload_file(local_model_path, self.bucket_name, s3_key)
            logging.info(f"Uploaded '{local_model_path}' to S3 as '{s3_key}'")
        except Exception as e:
            logging.error(f"Upload failed: {e}")

    def pull_model(self, s3_key, local_model_path=None):
        """Download model from S3"""
        if not local_model_path:
            local_model_path = os.path.join("models", os.path.basename(s3_key))
        try:
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            self.s3.download_file(self.bucket_name, s3_key, local_model_path)
            logging.info(f"Downloaded '{s3_key}' to '{local_model_path}'")
        except Exception as e:
            logging.error(f"Download failed: {e}")

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
                    logging.info(f"Deleted old model: {file}")
                except Exception as e:
                    logging.warning(f"Failed to delete {file}: {e}")
