"""Central registry for ASR model metadata shared across backend utilities."""

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

DEFAULT_CONFIG = {
    "USE_FLOAT16": True,
    "MAX_BUFFER_SECONDS": 10,
    "TARGET_SAMPLERATE": 16000,
    "VAD_THRESHOLD": 0.5,
    "SILENCE_DURATION_S": 1.0,
}

SUPPORTED_MODELS = {
    "GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned": {
        "label": "Sámi Parliament (CTC)",
        "language": "sme",
        "type": "ctc",
        "processor_cls": Wav2Vec2Processor,
        "model_cls": Wav2Vec2ForCTC,
        "prefer_float16": False,
    },
}
