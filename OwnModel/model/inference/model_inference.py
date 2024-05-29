import torch
from model import SpeechRecognition
from text_process import GreedyDecoder

def load_model(model_path, device):
    model = SpeechRecognition(hidden_size=1024, num_classes=29, n_feats=81, num_layers=1, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer(model, input_tensor, device):
    hidden = model._init_hidden(input_tensor.size(0))
    with torch.no_grad():
        output, _ = model(input_tensor, hidden)
    decoded_output, _ = GreedyDecoder(output.transpose(0, 1), None, None)
    return decoded_output[0]