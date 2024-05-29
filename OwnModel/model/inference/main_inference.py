import torch
from model_inference import load_model, infer
from preprocess_audio import preprocess_audio

def main(audio_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    input_tensor = preprocess_audio(audio_path, device)
    transcription = infer(model, input_tensor, device)
    print("Transcription:", transcription)

if __name__ == "__main__":
    audio_file = "/path/to/your/input/audio.mp3"  # Change this to your input audio file path
    model_file = "model.pth"  # Change this to your saved model file path
    main(audio_file, model_file)