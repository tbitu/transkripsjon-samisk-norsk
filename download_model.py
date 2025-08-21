# download_model.py
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# De samme konfigurasjonene som i main.py
MODEL_NAME = "NbAiLab/nb-whisper-large"
USE_FLOAT16 = True
torch_dtype = "float16" if USE_FLOAT16 else "float32"

print(f"Laster ned og cacher modell: {MODEL_NAME}")

# Laster ned og lagrer modellen i Docker-imagets cache
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)

print("Modell lastet ned og cachet vellykket.")
