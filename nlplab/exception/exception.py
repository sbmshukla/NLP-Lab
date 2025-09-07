import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nlplab.loggin.logger import logging


class CustomException(Exception):
    """
    Custom exception for network security tools.
    Captures filename and line number where the error occurred for better traceability.
    """

    def __init__(self, error_message: str, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (
            f"Error occurred in Python script: '{self.file_name}', "
            f"line {self.lineno}. Message: {self.error_message}"
        )


def handle_exception(e: Exception):
    """Shortcut to raise and log CustomException cleanly."""
    try:
        raise CustomException(error_message=str(e), error_details=sys)
    except CustomException as e:
        print(e)
        logging.error(e)


# Entry point for moudle testing
if __name__ == "__main__":
    try:
        print(1 / 0)
    except Exception as e:
        handle_exception(e)
