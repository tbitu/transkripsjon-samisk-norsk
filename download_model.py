# download_model.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Konfigurasjon
SUPPORTED_MODELS = [
    "NbAiLab/nb-whisper-large",
    "NbAiLab/whisper-large-sme",
]
USE_FLOAT16 = True
torch_dtype = torch.float16 if USE_FLOAT16 and torch.cuda.is_available() else torch.float32

# 1. Last ned Whisper-modeller
for model_name in SUPPORTED_MODELS:
    print(f"Laster ned og cacher Whisper-modell: {model_name}")
    WhisperProcessor.from_pretrained(model_name)
    WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch_dtype)
    print(f"{model_name} lastet ned og cachet vellykket.")

# 2. Last ned Silero VAD-modell
print("Laster ned og cacher Silero VAD-modell...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
print("VAD-modell lastet ned og cachet vellykket.")

print("\nAlle nødvendige modeller er lastet ned og cachet.")
print("Oversettelse håndteres nå av TartuNLP Translation API v2 i frontend.")
