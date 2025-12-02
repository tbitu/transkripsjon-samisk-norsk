# Før du kjører: installer/oppdater nødvendige biblioteker OG ffmpeg
# pip install -r requirements.txt
# System-krav: ffmpeg må være installert og tilgjengelig i PATH

from flask import Flask, send_from_directory, request, jsonify
from flask_sock import Sock
from flask_cors import CORS
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import gc 
import traceback
import queue
import threading
import subprocess
import time
from collections import deque

from model_registry import DEFAULT_CONFIG, SUPPORTED_MODELS
from simple_websocket.errors import ConnectionClosed


class ModelManager:
    def __init__(self, available_models, device_name, preferred_dtype):
        self.available_models = available_models
        self.device = device_name
        self._target_device = torch.device(device_name)
        self.torch_dtype = preferred_dtype
        self._lock = threading.Lock()
        self._active_model_name = None
        self._active_entry = None

    def preload(self, model_name):
        self.get(model_name)

    def get(self, model_name):
        if model_name not in self.available_models:
            raise ValueError(f"Ustøttet modell forespurt: {model_name}")
        return self._ensure_loaded(model_name)

    def unload(self):
        with self._lock:
            self._unload_current_locked()

    def current_model_name(self):
        with self._lock:
            return self._active_model_name

    def _ensure_loaded(self, model_name):
        with self._lock:
            if self._active_model_name == model_name and self._active_entry:
                return (
                    self._active_entry["processor"],
                    self._active_entry["model"],
                    self._active_entry["spec"],
                )
            self._load_model_locked(model_name)
            return (
                self._active_entry["processor"],
                self._active_entry["model"],
                self._active_entry["spec"],
            )

    def _load_model_locked(self, model_name):
        print(f"Laster inn modell: {model_name}...")
        self._unload_current_locked()
        spec = self.available_models[model_name]
        processor_cls = spec["processor_cls"]
        model_cls = spec["model_cls"]

        preferred_dtype = self.torch_dtype
        if spec.get("prefer_float16") is False:
            preferred_dtype = torch.float32

        processor = processor_cls.from_pretrained(model_name)
        model = model_cls.from_pretrained(model_name, torch_dtype=preferred_dtype)
        model = model.to(self._target_device)

        self._active_entry = {"processor": processor, "model": model, "spec": spec}
        self._active_model_name = model_name
        print(f"{model_name} ble lastet og plassert på {self._target_device}.")

    def _unload_current_locked(self):
        if not self._active_entry:
            return
        print(f"Laster ut modell: {self._active_model_name}")
        model = self._active_entry.get("model")
        processor = self._active_entry.get("processor")
        del model
        del processor
        self._active_entry = None
        self._active_model_name = None
        gc.collect()
        if self._target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Oppsett av Flask-applikasjon ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
whisper_executor = ThreadPoolExecutor(max_workers=2)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/preload-model', methods=['POST'])
def preload_model_http():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get('modelName')
    if not model_name or model_name not in SUPPORTED_MODELS:
        return jsonify({"error": "Ukjent modell"}), 400
    try:
        model_manager.preload(model_name)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"status": "ready", "modelName": model_name})

# --- Last inn modeller ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEFAULT_CONFIG["USE_FLOAT16"] and torch.cuda.is_available():
    torch_dtype = torch.float16
    print("Bruker float16 for økt ytelse.")
else:
    torch_dtype = torch.float32
    print("Bruker float32 for stabilitet.")

print(f"Bruker enhet: {device} med dtype: {torch_dtype}")
model_manager = ModelManager(SUPPORTED_MODELS, device, torch_dtype)
vad_model = None

try:
    model_manager.preload(DEFAULT_CONFIG['MODEL_NAME'])
    print("Laster inn VAD-modell...")
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    print("VAD-modell lastet inn vellykket!")

except Exception as e:
    print(f"ADVARSEL: Kunne ikke laste modeller: {e}")
    traceback.print_exc()

# --- Hjelpefunksjoner ---
def process_audio_task(audio_input, config, send_fn):
    """Transkriberer lydklipp med valgt modell og sender tekst."""
    model_name = config.get('MODEL_NAME', DEFAULT_CONFIG['MODEL_NAME'])
    try:
        processor, active_model, spec = model_manager.get(model_name)
    except ValueError as exc:
        send_fn({"error": str(exc)})
        return
    try:
        with torch.no_grad():
            print(f"Prosesserer {len(audio_input) / config['TARGET_SAMPLERATE']:.2f} sekunder med lyd...")
            if spec['type'] == 'whisper':
                processed_input = processor(audio_input, sampling_rate=config['TARGET_SAMPLERATE'], return_tensors="pt")
                input_features = processed_input.input_features.to(device, dtype=torch_dtype)
                attention_mask = processed_input.attention_mask.to(device) if "attention_mask" in processed_input else torch.ones_like(input_features, device=device)

                language_code = spec.get('language', 'no')
                generation_kwargs = {
                    "attention_mask": attention_mask,
                    "task": "transcribe"
                }
                if language_code:
                    generation_kwargs["language"] = language_code

                predicted_ids = active_model.generate(input_features, **generation_kwargs)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            else:
                processed_input = processor(audio_input, sampling_rate=config['TARGET_SAMPLERATE'], return_tensors="pt", padding="longest")
                input_values = processed_input.input_values.to(device)
                attention_mask = processed_input.attention_mask.to(device) if "attention_mask" in processed_input else None
                logits = active_model(input_values, attention_mask=attention_mask).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0].strip()

            if transcription:
                print(f"Transkribert: {transcription}")
                # Send norsk tekst
                send_fn({
                    "type": "transcription",
                    "text": transcription,
                    "modelName": model_name
                })
            else:
                print("Ingen tekst gjenkjent.")
    except Exception:
        print("En uventet feil oppstod under transkribering:")
        traceback.print_exc() 
        send_fn({"error": "Internal server error during transcription."})
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Arbeider-tråder (uendret) ---
PRE_TRIGGER_CHUNKS = 6  # keep ~200ms of audio before the trigger fires
def ffmpeg_reader_thread(ffmpeg_proc, pcm_queue):
    while True:
        chunk = ffmpeg_proc.stdout.read(512 * 2) 
        if not chunk: break
        pcm_queue.put(chunk)
    pcm_queue.put(None)

def pcm_processor_worker(pcm_queue, send_fn, config, config_lock):
    speech_buffer = bytearray()
    triggered = False
    last_speech_time = 0
    pre_trigger_queue = deque(maxlen=PRE_TRIGGER_CHUNKS)
    
    print("VAD-prosessor startet.")
    while True:
        chunk = pcm_queue.get()
        if chunk is None:
            if speech_buffer:
                with config_lock: current_config = config.copy()
                audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                whisper_executor.submit(process_audio_task, audio_input, current_config, send_fn)
            break
        
        with config_lock: current_config = config.copy()
        audio_chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_chunk_torch = torch.from_numpy(audio_chunk_np)
        speech_prob = vad_model(audio_chunk_torch, current_config['TARGET_SAMPLERATE']).item()
        
        if speech_prob > current_config['VAD_THRESHOLD']:
            if not triggered:
                print("Stemme oppdaget, starter opptak av segment...")
                triggered = True
                if pre_trigger_queue:
                    for pre_chunk in pre_trigger_queue:
                        speech_buffer.extend(pre_chunk)
                    pre_trigger_queue.clear()
            speech_buffer.extend(chunk)
            last_speech_time = time.time()
        elif triggered:
            speech_buffer.extend(chunk)
            if time.time() - last_speech_time > current_config['SILENCE_DURATION_S']:
                print(f"Trigger: Stillhet i {current_config['SILENCE_DURATION_S']}s etter tale.")
                audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                whisper_executor.submit(process_audio_task, audio_input, current_config, send_fn)
                speech_buffer.clear()
                triggered = False
                continue
        else:
            pre_trigger_queue.append(chunk)

        
        if triggered and len(speech_buffer) / (current_config['TARGET_SAMPLERATE'] * 2) >= current_config['MAX_BUFFER_SECONDS']:
            print(f"Trigger: Maks varighet på {current_config['MAX_BUFFER_SECONDS']}s nådd.")
            audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            whisper_executor.submit(process_audio_task, audio_input, current_config, send_fn)
            speech_buffer.clear()
            triggered = False

# --- WebSocket-endepunkt ---
@sock.route('/stream')
def stream(ws):
    print("Klient koblet til, starter vedvarende ffmpeg-prosess.")
    session_config = DEFAULT_CONFIG.copy()
    config_lock = threading.Lock()
    loader_state_lock = threading.Lock()
    model_loader_state = {"thread": None, "pending": None}
    ws_send_lock = threading.Lock()

    def safe_ws_send(payload):
        try:
            with ws_send_lock:
                ws.send(json.dumps(payload))
        except ConnectionClosed as closed_err:
            print(f"WebSocket lukket, dropper melding: {closed_err}")
        except Exception as send_error:
            print(f"Kunne ikke sende WS-melding: {send_error}")

    def start_model_load(target_model_name):
        def _load(name):
            try:
                model_manager.preload(name)
            except Exception as load_error:
                print(f"Kunne ikke laste modell {name}: {load_error}")
                traceback.print_exc()
                safe_ws_send({
                    "type": "model_status",
                    "status": "error",
                    "modelName": name,
                    "error": str(load_error)
                })
            else:
                with config_lock:
                    session_config['MODEL_NAME'] = name
                print(f"Oppdaterer konfigurasjon: MODEL_NAME = {name}")
                safe_ws_send({
                    "type": "model_status",
                    "status": "ready",
                    "modelName": name
                })
            finally:
                with loader_state_lock:
                    next_name = model_loader_state["pending"]
                    model_loader_state["pending"] = None
                    model_loader_state["thread"] = None
                if next_name and next_name != name:
                    safe_ws_send({
                        "type": "model_status",
                        "status": "loading",
                        "modelName": next_name
                    })
                    start_model_load(next_name)

        with loader_state_lock:
            existing_thread = model_loader_state["thread"]
            if existing_thread and existing_thread.is_alive():
                model_loader_state["pending"] = target_model_name
                return
            thread = threading.Thread(target=_load, args=(target_model_name,), daemon=True)
            model_loader_state["thread"] = thread
            thread.start()

    command = ['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-f', 's16le', '-ac', '1', '-ar', str(session_config['TARGET_SAMPLERATE']), 'pipe:1']
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pcm_queue = queue.Queue()
    reader = threading.Thread(target=ffmpeg_reader_thread, args=(ffmpeg_process, pcm_queue))
    processor_thread = threading.Thread(target=pcm_processor_worker, args=(pcm_queue, safe_ws_send, session_config, config_lock))
    reader.start()
    processor_thread.start()
    try:
        while True:
            message = ws.receive()
            if message is None: break
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get('type') == 'config':
                        key, value = data.get('key'), data.get('value')
                        if key == 'MODEL_NAME':
                            if value in SUPPORTED_MODELS:
                                with config_lock:
                                    current_model = session_config.get('MODEL_NAME')
                                if current_model == value and model_manager.current_model_name() == value:
                                    safe_ws_send({
                                        "type": "model_status",
                                        "status": "ready",
                                        "modelName": value
                                    })
                                    continue
                                safe_ws_send({
                                    "type": "model_status",
                                    "status": "loading",
                                    "modelName": value
                                })
                                start_model_load(value)
                            else:
                                print(f"Ignorerer ukjent modellvalg: {value}")
                        elif key in session_config:
                            try:
                                numeric_value = float(value)
                            except (TypeError, ValueError):
                                print(f"Kunne ikke parse numerisk verdi for {key}: {value}")
                                continue
                            with config_lock:
                                session_config[key] = numeric_value
                            print(f"Oppdaterer konfigurasjon: {key} = {numeric_value}")
                except json.JSONDecodeError:
                    print("Mottok ugyldig JSON-melding.")
            else:
                try:
                    ffmpeg_process.stdin.write(message)
                except BrokenPipeError:
                    print("FFmpeg-prosess avsluttet uventet.")
                    break
    except Exception as e:
        print(f"En feil oppstod i WebSocket-tilkoblingen: {e}")
    finally:
        print("Klient koblet fra. Lukker ffmpeg og rydder opp tråder.")
        if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        try:
            ffmpeg_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
        if reader.is_alive(): pcm_queue.put(None)
        reader.join(timeout=2)
        processor_thread.join(timeout=2)
        print("Opprydding fullført.")

if __name__ == '__main__':
    print(f"Starter tjener på http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
