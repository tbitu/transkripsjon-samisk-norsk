# download_model.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Konfigurasjon
WHISPER_MODEL = "NbAiLab/nb-whisper-large"
USE_FLOAT16 = True
torch_dtype = "float16" if USE_FLOAT16 else "float32"

# 1. Last ned Whisper-modell
print(f"Laster ned og cacher Whisper-modell: {WHISPER_MODEL}")
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, torch_dtype=torch_dtype)
print("Whisper-modell lastet ned og cachet vellykket.")

# 2. Last ned Silero VAD-modell
print("Laster ned og cacher Silero VAD-modell...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
print("VAD-modell lastet ned og cachet vellykket.")

print("\nAlle nødvendige modeller er lastet ned og cachet.")
print("Oversettelse håndteres nå av TartuNLP Translation API v2 i frontend.")
