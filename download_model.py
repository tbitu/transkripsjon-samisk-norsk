# download_model.py
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from model_registry import SUPPORTED_MODELS
USE_FLOAT16 = True
torch_dtype = torch.float16 if USE_FLOAT16 and torch.cuda.is_available() else torch.float32

# 1. Last ned modeller
for model_name, spec in SUPPORTED_MODELS.items():
    model_type = spec.get("type", "whisper")
    print(f"Laster ned og cacher {model_type.upper()}-modell: {model_name}")
    if model_type == "whisper":
        WhisperProcessor.from_pretrained(model_name)
        WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        Wav2Vec2Processor.from_pretrained(model_name)
        Wav2Vec2ForCTC.from_pretrained(model_name)
    print(f"{model_name} lastet ned og cachet vellykket.")

# 2. Last ned Silero VAD-modell
print("Laster ned og cacher Silero VAD-modell...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
print("VAD-modell lastet ned og cachet vellykket.")

print("\nAlle nødvendige modeller er lastet ned og cachet.")
print("Oversettelse håndteres nå av TartuNLP Translation API v2 i frontend.")
