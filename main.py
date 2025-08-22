# Før du kjører: installer/oppdater nødvendige biblioteker OG ffmpeg
# pip install -r requirements.txt
# System-krav: ffmpeg må være installert og tilgjengelig i PATH

from flask import Flask, send_from_directory
from flask_sock import Sock
from flask_cors import CORS
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import json
import gc 
import traceback
import queue
import threading
import subprocess
import time

# --- Standard Konfigurasjon ---
DEFAULT_CONFIG = {
    "MODEL_NAME": "NbAiLab/nb-whisper-large",
    "USE_FLOAT16": True,
    "MAX_BUFFER_SECONDS": 10,
    "TARGET_SAMPLERATE": 16000,
    # VAD-spesifikke innstillinger
    "VAD_THRESHOLD": 0.5, # Følsomhet for VAD. Høyere = mindre følsom.
    "SILENCE_DURATION_S": 1.0, # Hvor lenge det må være stille etter tale før sending.
}

# --- Oppsett av Flask-applikasjon ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
whisper_executor = ThreadPoolExecutor(max_workers=1) 

# Funksjon for å servere HTML-klienten
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# --- Last inn modeller ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEFAULT_CONFIG["USE_FLOAT16"] and torch.cuda.is_available():
    torch_dtype = torch.float16
    print("Bruker float16 for økt ytelse.")
else:
    torch_dtype = torch.float32
    print("Bruker float32 for stabilitet.")

print(f"Bruker enhet: {device} med dtype: {torch_dtype}")
processor = None
model = None
vad_model = None
try:
    print(f"Laster inn modell: {DEFAULT_CONFIG['MODEL_NAME']}...")
    processor = WhisperProcessor.from_pretrained(DEFAULT_CONFIG['MODEL_NAME'])
    model = WhisperForConditionalGeneration.from_pretrained(DEFAULT_CONFIG['MODEL_NAME'], torch_dtype=torch_dtype).to(device)
    print("Modell og prosessor ble lastet inn vellykket!")
    
    # **NYTT: Initialiser Silero VAD-modellen**
    print("Laster inn VAD-modell...")
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    print("VAD-modell lastet inn vellykket!")

except Exception as e:
    print(f"ADVARSEL: Kunne ikke laste modeller: {e}")
    traceback.print_exc()

# --- Hjelpefunksjoner ---
def process_audio_task(audio_input, ws, config):
    if not model or not processor:
        ws.send(json.dumps({"error": "Modellen er ikke tilgjengelig"}))
        return
    try:
        with torch.no_grad():
            print(f"Prosesserer {len(audio_input) / config['TARGET_SAMPLERATE']:.2f} sekunder med lyd...")
            
            processed_input = processor(audio_input, sampling_rate=config['TARGET_SAMPLERATE'], return_tensors="pt")
            input_features = processed_input.input_features.to(device, dtype=torch_dtype)
            
            if "attention_mask" in processed_input:
                attention_mask = processed_input.attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_features, device=device)

            predicted_ids = model.generate(input_features, attention_mask=attention_mask, language="no", task="transcribe")
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            if transcription:
                print(f"Transkribert: {transcription}")
                ws.send(json.dumps({"transcription": transcription}))
            else:
                print("Ingen tekst gjenkjent.")
    except Exception as e:
        print("En uventet feil oppstod under transkribering:")
        traceback.print_exc() 
        ws.send(json.dumps({"error": "Internal server error during transcription."}))
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- Arbeider-tråder ---
def ffmpeg_reader_thread(ffmpeg_proc, pcm_queue):
    while True:
        # Les små biter for rask VAD-respons (32ms)
        chunk = ffmpeg_proc.stdout.read(512 * 2) 
        if not chunk:
            break
        pcm_queue.put(chunk)
    pcm_queue.put(None)

def pcm_processor_worker(pcm_queue, ws, config, config_lock):
    """**Omskrevet pcm_processor_worker med VAD**"""
    speech_buffer = bytearray()
    triggered = False
    last_speech_time = 0
    
    print("VAD-prosessor startet.")
    while True:
        chunk = pcm_queue.get()
        if chunk is None:
            if speech_buffer:
                with config_lock:
                    current_config = config.copy()
                audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                whisper_executor.submit(process_audio_task, audio_input, ws, current_config)
            break
        
        with config_lock:
            current_config = config.copy()

        audio_chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_chunk_torch = torch.from_numpy(audio_chunk_np)
        
        speech_prob = vad_model(audio_chunk_torch, current_config['TARGET_SAMPLERATE']).item()
        
        if speech_prob > current_config['VAD_THRESHOLD']:
            if not triggered:
                print("Stemme oppdaget, starter opptak av segment...")
                triggered = True
            speech_buffer.extend(chunk)
            last_speech_time = time.time()
        elif triggered:
            speech_buffer.extend(chunk)
            if time.time() - last_speech_time > current_config['SILENCE_DURATION_S']:
                print(f"Trigger: Stillhet i {current_config['SILENCE_DURATION_S']}s etter tale.")
                audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                whisper_executor.submit(process_audio_task, audio_input, ws, current_config)
                speech_buffer.clear()
                triggered = False
        
        if triggered and len(speech_buffer) / (current_config['TARGET_SAMPLERATE'] * 2) >= current_config['MAX_BUFFER_SECONDS']:
            print(f"Trigger: Maks varighet på {current_config['MAX_BUFFER_SECONDS']}s nådd.")
            audio_input = np.frombuffer(speech_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            whisper_executor.submit(process_audio_task, audio_input, ws, current_config)
            speech_buffer.clear()
            triggered = False

# --- WebSocket-endepunkt ---
@sock.route('/stream')
def stream(ws):
    print("Klient koblet til, starter vedvarende ffmpeg-prosess.")
    
    session_config = DEFAULT_CONFIG.copy()
    config_lock = threading.Lock()

    command = [
        'ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-f', 's16le', '-ac', '1',
        '-ar', str(session_config['TARGET_SAMPLERATE']), 'pipe:1'
    ]
    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    pcm_queue = queue.Queue()
    reader = threading.Thread(target=ffmpeg_reader_thread, args=(ffmpeg_process, pcm_queue))
    processor = threading.Thread(target=pcm_processor_worker, args=(pcm_queue, ws, session_config, config_lock))
    reader.start()
    processor.start()

    try:
        while True:
            message = ws.receive()
            if message is None: break
            
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get('type') == 'config':
                        with config_lock:
                            key, value = data.get('key'), data.get('value')
                            if key in session_config:
                                print(f"Oppdaterer konfigurasjon: {key} = {value}")
                                session_config[key] = value
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
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        try:
            ffmpeg_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
        
        if reader.is_alive():
            pcm_queue.put(None)
        
        reader.join(timeout=2)
        processor.join(timeout=2)
        print("Opprydding fullført.")

if __name__ == '__main__':
    print(f"Starter tjener på http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
