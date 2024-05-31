#errors.py
from flask import jsonify

class InvalidFileError(Exception):
    def __init__(self, message="Invalid file type"):
        self.message = message
        super().__init__(self.message)

class NoFileProvidedError(Exception):
    def __init__(self, message="No file provided"):
        self.message = message
        super().__init__(self.message)

class TranscriptionError(Exception):
    def __init__(self, message="Transcription failed due to server error"):
        self.message = message
        super().__init__(self.message)

def handle_invalid_file_error(e):
    response = jsonify({"error": str(e)})
    response.status_code = 400
    return response

def handle_no_file_provided_error(e):
    response = jsonify({"error": str(e)})
    response.status_code = 400
    return response

def handle_transcription_error(e):
    response = jsonify({"error": str(e)})
    response.status_code = 500
    return response