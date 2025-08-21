# Før du kjører: installer/oppdater nødvendige biblioteker OG ffmpeg
# pip install --upgrade Flask flask_cors transformers torch torchaudio soundfile flask-sock numpy librosa
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

# --- Standard Konfigurasjon ---
# Disse verdiene brukes som standard, men kan overstyres av klienten
DEFAULT_CONFIG = {
    "MODEL_NAME": "NbAiLab/nb-whisper-large",
    "USE_FLOAT16": True,
    "MAX_BUFFER_SECONDS": 10,
    "TARGET_SAMPLERATE": 16000,
    "SILENCE_THRESHOLD": 0.04,
    "SILENCE_DURATION_S": 1.5,
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

# --- Last inn Whisper-modellen ---
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
try:
    print(f"Laster inn modell: {DEFAULT_CONFIG['MODEL_NAME']}...")
    processor = WhisperProcessor.from_pretrained(DEFAULT_CONFIG['MODEL_NAME'])
    model = WhisperForConditionalGeneration.from_pretrained(DEFAULT_CONFIG['MODEL_NAME'], torch_dtype=torch_dtype).to(device)
    print("Modell og prosessor ble lastet inn vellykket!")
except Exception as e:
    print(f"ADVARSEL: Kunne ikke laste modellen: {e}")
    traceback.print_exc()

# --- Hjelpefunksjoner ---
def is_silent(audio_data, samplerate, config):
    samples_to_check = int(min(samplerate * config["SILENCE_DURATION_S"], len(audio_data)))
    if samples_to_check < 1024: return False
    segment = audio_data[-samples_to_check:]
    rms = np.sqrt(np.mean(segment**2))
    print(f"Sjekker stillhet... RMS: {rms:.4f} (Terskel: {config['SILENCE_THRESHOLD']})")
    return rms < config["SILENCE_THRESHOLD"]

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
        chunk = ffmpeg_proc.stdout.read(int(DEFAULT_CONFIG['TARGET_SAMPLERATE'] * 2 * 0.5))
        if not chunk:
            break
        pcm_queue.put(chunk)
    pcm_queue.put(None)

def pcm_processor_worker(pcm_queue, ws, config, config_lock):
    pcm_buffer = bytearray()
    
    while True:
        chunk = pcm_queue.get()
        if chunk is None:
            if pcm_buffer:
                with config_lock:
                    current_config = config.copy()
                audio_input = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                whisper_executor.submit(process_audio_task, audio_input, ws, current_config)
            break
        
        pcm_buffer.extend(chunk)
        
        with config_lock:
            current_config = config.copy()
        
        target_bytes = current_config['TARGET_SAMPLERATE'] * 2 * current_config['MAX_BUFFER_SECONDS']
        should_process = False
        
        audio_for_analysis = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(pcm_buffer) >= target_bytes:
            print(f"Trigger: Maks varighet ({len(audio_for_analysis)/current_config['TARGET_SAMPLERATE']:.2f}s) nådd.")
            should_process = True
        elif is_silent(audio_for_analysis, current_config['TARGET_SAMPLERATE'], current_config):
            if len(audio_for_analysis) > current_config['TARGET_SAMPLERATE'] * 1.0:
                 print("Trigger: Stillhet oppdaget.")
                 should_process = True

        if should_process:
            overall_rms = np.sqrt(np.mean(audio_for_analysis**2))
            if overall_rms < current_config['SILENCE_THRESHOLD']:
                print(f"Avviser klipp, sannsynligvis bare støy (Overall RMS: {overall_rms:.4f})")
            else:
                whisper_executor.submit(process_audio_task, audio_for_analysis, ws, current_config)
            
            pcm_buffer.clear()

# --- WebSocket-endepunkt ---
@sock.route('/stream')
def stream(ws):
    print("Klient koblet til, starter vedvarende ffmpeg-prosess.")
    
    # **NYTT: Trådsikker konfigurasjon for denne tilkoblingen**
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
            
            # Sjekk om meldingen er tekst (JSON) eller binærdata (lyd)
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
                # Dette er lyd-data
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
    print(f"Starter tjener på http://0._0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
