# Copilot Instructions

## Big Picture
- Realtime STT+translation stack: Flask backend in [main.py](../main.py) handles audio ingestion + Whisper inference, while [index.html](../index.html) provides the UI, streams microphone audio over WebSocket, and calls the TartuNLP Translation API for target-language text.
- Audio arrives as MediaRecorder chunks → `/stream` WebSocket → ffmpeg (convert to mono 16k PCM) → Silero VAD gating → Whisper inference via `process_audio_task()`. Translation never touches the backend.
- Only two Whisper checkpoints are supported (`NbAiLab/nb-whisper-large`, `NbAiLab/whisper-large-sme`). VRAM is the bottleneck, so `ModelManager` keeps exactly one model on the selected device and hot-swaps on demand.

## Backend Implementation Patterns
- `ModelManager` in [main.py](../main.py) centralizes processor/model loading, locking, and CUDA cache cleanup. Always request ramps through `get()`/`preload()` instead of instantiating Whisper classes directly.
- Voice activity handling lives in `pcm_processor_worker()` using Silero VAD to decide when to ship a buffered segment to Whisper and when to enforce `MAX_BUFFER_SECONDS`/`SILENCE_DURATION_S` limits. Respect these config keys when adding new behavior so UI sliders keep working.
- The WebSocket handler accepts two message types: JSON `config` messages (update `session_config` or trigger model swap) and binary audio frames. Preserve this contract if you add new client messages.
- Cleaning up GPU memory is critical—mirror the existing `gc.collect()` + `torch.cuda.empty_cache()` pattern when creating long-lived tensors or background tasks.

## Frontend Expectations
- The single-page UI in [index.html](../index.html) is framework-free Tailwind. New UI work should follow the existing utility-class style and avoid build steps.
- Translation uses `TARTUNLP_API_URL` with `{text, src, tgt, domain}` payloads. Source language derives from the selected Whisper model (`resolveSourceLanguage()`); keep the mapping in sync with backend `SUPPORTED_MODELS`.
- Model selection, slider controls, and "Norsk/Oversettelse/Begge" buttons all manipulate DOM directly. When adding controls, wire them through the same `sendConfig()` → backend config path so behavior remains centralized.

## Local Development Workflow
- GPU path: install ffmpeg, create venv, run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129`, then `pip install -r requirements.txt`, and start via `python main.py` (see [README.md](../README.md)).
- CPU fallback works but is slow; set `DEFAULT_CONFIG["USE_FLOAT16"] = False` if you hit dtype issues on non-CUDA hardware.
- [download_model.py](../download_model.py) pre-caches Whisper + Silero models (also invoked in Docker build). Run it manually before travel/offline demos to avoid runtime downloads.
- For Windows + WSL + GPU, use the guided script [wsl-run-transkripsjon.ps1](../wsl-run-transkripsjon.ps1) which enforces driver, WSL2, and Docker prerequisites before starting the container.

## Docker & CI
- Docker image builds off `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`, installs PyTorch with CUDA 12.9 wheels first, then the remaining requirements, then pre-downloads models. Keep this ordering to leverage layer caching.
- The workflow [workflows/docker-publish.yml](workflows/docker-publish.yml) runs on pushes to `main`, frees disk space, logs into GHCR with `CR_PAT`, and tags `ghcr.io/<owner>/transkripsjon-samisk-norsk:latest`. Update secrets/registry references there when changing release targets.

## Conventions & Tips
- Stick to ASCII in source files unless the file already contains Norwegian text (the repo mixes both, but backend code comments are already Norwegian).
- Favor threads and queues over async; the backend already uses `ThreadPoolExecutor` for Whisper inference and raw `threading.Thread` elsewhere.
- Logging is currently via `print()`; if you add structured logging, ensure it still appears in container stdout so GHCR users can tail `docker logs`.
- Avoid blocking operations in the WebSocket receive loop; move heavy work to the executor or PCM worker to keep audio ingestion responsive.
