import logging
import os
from datetime import datetime

# Generate a unique log filename using current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path where logs will be stored, including the log filename
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory path if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Log level
)

# Entry point for moudle testing
if __name__ == "__main__":
    logging.info("Logging system initialized successfully. Ready to capture events.")
