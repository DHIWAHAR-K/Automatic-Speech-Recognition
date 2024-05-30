# app.py
from logger import server_logger
from transcribe import transcribe_audio
from flask import Flask, request, jsonify
from errors import InvalidFileError, NoFileProvidedError, TranscriptionError, handle_invalid_file_error, handle_no_file_provided_error, handle_transcription_error

app = Flask(__name__)

app.register_error_handler(InvalidFileError, handle_invalid_file_error)
app.register_error_handler(NoFileProvidedError, handle_no_file_provided_error)
app.register_error_handler(TranscriptionError, handle_transcription_error)

@app.route('/transcribe', methods=['POST'])
def transcribe_endpoint():
    server_logger.info('Route /transcribe hit')
    if 'file' not in request.files:
        raise NoFileProvidedError()
    
    file = request.files['file']
    if file.filename == '':
        raise InvalidFileError("Empty file name")

    if not allowed_file(file.filename):
        raise InvalidFileError()
    
    language = request.form.get('language')
    return transcribe_audio(file, language)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(port=8080)