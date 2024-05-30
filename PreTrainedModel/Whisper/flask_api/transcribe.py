# transcribe.py
import os
import whisper
import logging
from flask import jsonify
from logger import server_logger
from model_config import MODEL_NAME
from errors import TranscriptionError

# Load the Whisper model based on configuration
model = whisper.load_model(MODEL_NAME)

def transcribe_audio(file, language=None):
    audio_file_path = None
    try:
        # Save the audio file temporarily
        audio_file_path = f"temp_audio_file.{file.filename.split('.')[-1]}"
        file.save(audio_file_path)

        # Transcribe the audio file using Whisper
        result = model.transcribe(audio_file_path, language=language)

        # Clean up temporary file
        os.remove(audio_file_path)

        # Log the transcription result
        transcription_text = result["text"]
        server_logger.info(f"Transcription result: {transcription_text}")

        # Return the transcription result
        return jsonify({"transcription": transcription_text})
    
    except Exception as e:
        server_logger.error(f"Error during transcription: {str(e)}")
        raise TranscriptionError()
    finally:
        if audio_file_path and os.path.exists(audio_file_path):
            os.remove(audio_file_path)

# # transcribe.py
# import os
# from flask import jsonify
# from logger import server_logger
# from model_config import MODEL_NAME
# from faster_whisper import WhisperModel
# from errors import TranscriptionError

# # Load the Faster Whisper model based on configuration
# try:
#     model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
#     server_logger.info(f"Faster Whisper model '{MODEL_NAME}' loaded successfully.")
# except Exception as e:
#     server_logger.error(f"Error loading Faster Whisper model: {str(e)}")
#     raise TranscriptionError("Failed to load the transcription model")

# def transcribe_audio(file, language=None):
#     audio_file_path = None
#     try:
#         # Save the audio file temporarily
#         audio_file_path = f"temp_audio_file.{file.filename.split('.')[-1]}"
#         file.save(audio_file_path)

#         # Transcribe the audio file using Faster Whisper
#         segments, info = model.transcribe(audio_file_path, beam_size=5, language=language, condition_on_previous_text=False)

#         # Concatenate all segments into a single transcription text
#         transcription_text = " ".join([segment.text for segment in segments])

#         # Clean up temporary file
#         os.remove(audio_file_path)

#         # Log the transcription result
#         server_logger.info(f"Transcription result: {transcription_text}")

#         # Return the transcription result
#         return jsonify({"transcription": transcription_text})
    
#     except Exception as e:
#         server_logger.error(f"Error during transcription: {str(e)}")
#         raise TranscriptionError()
#     finally:
#         if audio_file_path and os.path.exists(audio_file_path):
#             os.remove(audio_file_path)