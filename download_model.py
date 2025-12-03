# download_model.py
import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from diart import models as diart_models

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

pyannote_token = (
    os.getenv("PYANNOTE_TOKEN")
    or os.getenv("PYANNOTE_AUTH_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HF_TOKEN")
)

try:
    print("Laster ned og cacher pyannote-modeller for diariseringspipeline...")
    diart_models.SegmentationModel.from_pyannote(
        "pyannote/segmentation", use_hf_token=pyannote_token or True
    ).load()
    diart_models.EmbeddingModel.from_pyannote(
        "pyannote/embedding", use_hf_token=pyannote_token or True
    ).load()
    print("Pyannote-modeller lastet ned og cachet vellykket.")
except Exception as pyannote_error:
    print(f"ADVARSEL: Klarte ikke å laste pyannote-modeller: {pyannote_error}")

print("\nAlle nødvendige modeller er lastet ned og cachet.")
print("Oversettelse håndteres nå av TartuNLP Translation API v2 i frontend.")
