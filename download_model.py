# download_model.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSeq2SeqLM, NllbTokenizer

# Konfigurasjon
WHISPER_MODEL = "NbAiLab/nb-whisper-large"
# RETTET: Bytter tilbake til Facebook/Meta AI sin NLLB-modell
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
USE_FLOAT16 = True
torch_dtype = "float16" if USE_FLOAT16 else "float32"

# 1. Last ned Whisper-modell
print(f"Laster ned og cacher Whisper-modell: {WHISPER_MODEL}")
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, torch_dtype=torch_dtype)
print("Whisper-modell lastet ned og cachet vellykket.")

# 2. Last ned Oversettelsesmodell (NLLB)
print(f"Laster ned og cacher oversettelsesmodell: {TRANSLATION_MODEL}")
# Bruker NllbTokenizer og AutoModelForSeq2SeqLM for denne modellen
translation_tokenizer = NllbTokenizer.from_pretrained(TRANSLATION_MODEL)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
print("Oversettelsesmodell lastet ned og cachet vellykket.")

# 3. Last ned Silero VAD-modell
print("Laster ned og cacher Silero VAD-modell...")
torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)
print("VAD-modell lastet ned og cachet vellykket.")
