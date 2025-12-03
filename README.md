# Sanntidstranskripsjon og oversettelse

Dette prosjektet er en sanntids-simultanoversetter som transkriberer samisk og norsk tale til norsk tekst, og deretter oversetter teksten til valgfritt målspråk. Løsningen er bygget med en Python-backend for transkripsjon og et webgrensesnitt som håndterer oversettelse. I denne forken er backend-en utvidet med diarisering (hvem snakker når) via `diart` + `pyannote.audio`, slik at hver tekstbit vises med egen farge i UI-et. Teknologistakken består av:
- **Transkripsjon (backend):** `NbAiLab/nb-whisper-large`, `NbAiLab/whisper-large-sme` og `GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned`
- **Diarisering (backend):** `diart` (`SpeakerDiarization`) + `pyannote/segmentation` og `pyannote/embedding` (krever Hugging Face-token)
- **Oversettelse (frontend):** TartuNLP Translation API v2 for norsk til diverse uralske og samiske språk

## Støttede oversettelser

Applikasjonen kan oversette fra norsk til følgende språk:
- **Samiske språk:** nordsamisk, sørsamisk, lulesamisk, enaresamisk, skoltesamisk
- **Uralske språk:** karelsk, livvi, vepsisk, livisk, khanti, mansi, komi-permjakisk, komi-syrjensk, udmurtisk, austmarisk, vestmarisk, erzja, moksja, lüüdisk, võro

## Tilgjengelige ASR-modeller

| Modell | Type | Typisk bruk |
| --- | --- | --- |
| `NbAiLab/nb-whisper-large` | Whisper seq2seq | Gir norsk tekst direkte fra samisk/norsk tale (standardvalg). |
| `NbAiLab/whisper-large-sme` | Whisper seq2seq | Transkriberer til nordsamisk tekst uten oversettelse. |
| `GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned` | Wav2Vec2 CTC | Lettere modell trent på Sametinget-opptak, leverer nordsamisk tekst (CTC). |

Alle modellene krever mono PCM 16 kHz-lyd. Wav2Vec2-modellen kjører alltid i float32; på GPU bruker den mindre VRAM enn Whisper Large, men lastetid på CPU kan være lengre.

## Diarisering og fargekoding

- **diart + pyannote:** Serveren kjører en kontinuerlig `SpeakerDiarization`-pipeline (5 s vindu / 0,5 s steg) og sender resultatet videre til samme 2 s vindu som Whisper/Wav2Vec2 transkriberer. Hver bit tekst blir dermed tagget med `speakerId`, start- og sluttid.
- **Frontend-oppdatering:** Brukergrensesnittet viser nå segmentene i kronologisk rekkefølge med konsistente farger per taler. Oversettelsen følger samme segmentstruktur slik at målteksten henger sammen med riktig taler.
- **Krav:** Pyannote-modellene er lisensiert. Sørg for å [logge inn hos Hugging Face](https://huggingface.co/settings/tokens) og akseptere vilkårene for `pyannote/segmentation` og `pyannote/embedding`. Sett deretter miljøvariabelen `PYANNOTE_TOKEN` (eller logg inn med `huggingface-cli login`) før du starter serveren.
- **Ressursbruk:** diariseringsmodellene legger til ca. 3–4 GB VRAM. Hvis GPU-en er marginal, kan du kjøre diarisering på CPU ved å sette `CUDA_VISIBLE_DEVICES=` før oppstart, men forvent høyere latens.

## Bruk med Docker (anbefalt)

Du kan enkelt kjøre prosjektet med ferdigbygd Docker-image fra GitHub Container Registry (GHCR). Dette gir deg korrekt miljø for PyTorch og GPU-støtte uten å installere Python og avhengigheter manuelt.

> ⚠️ Det finnes også et hjelpeskript for Windows/WSL ([wsl-run-transkripsjon.ps1](wsl-run-transkripsjon.ps1)), men det er utestet/ufullført og bør brukes kun som veiledning.

### 1. Kjør med ferdigbygd image fra GHCR

**Forutsetninger:**
- Docker installert på systemet
- NVIDIA GPU og [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installert for GPU-akselerasjon

**Kommando:**

```bash
docker run --gpus all -p 5000:5000 ghcr.io/tbitu/transkripsjon-samisk-norsk:latest
```

- `--gpus all`: Gir Docker-containeren tilgang til alle NVIDIA-GPUer
- `-p 5000:5000`: Eksponerer webgrensesnittet på port 5000

**Bruk**:  
Når containeren kjører, åpne [http://localhost:5000](http://localhost:5000) i nettleseren, eller bruk din lokale IP-adresse på andre enheter i nettverket.

---

### 2. Bygg Docker-image selv (for utviklere/avanserte brukere)

#### a. Klon repositoryet

```bash
git clone https://github.com/tbitu/transkripsjon-samisk-norsk.git
cd transkripsjon-samisk-norsk
```

#### b. Bygg Docker-image

Imaget er satt opp for NVIDIA GPU og inkluderer PyTorch med CUDA-støtte.

```bash
docker build -t transkripsjon-samisk-norsk .
```

#### c. Kjør image med GPU-tilgang

```bash
docker run --gpus all -p 5000:5000 transkripsjon-samisk-norsk
```

---

## 3. Kjøre uten Docker (manuelt)

**Forutsetninger:**

1. **Python 3.8+**
2. **ffmpeg** installert og i PATH:
    - Ubuntu/Debian: `sudo apt install ffmpeg`
    - macOS: `brew install ffmpeg`
    - Windows: Last ned fra [ffmpeg.org](https://ffmpeg.org/download.html) og legg til i PATH
3. **CUDA-kompatibel GPU (anbefalt):** For å bruke large-modellen trengs ca. 10–12 GB VRAM. Det går på CPU, men blir mye tregere.
4. **Internettilkobling:** Nødvendig for oversettelse via TartuNLP API
5. **Hugging Face-token for pyannote-modeller:** Logg inn med `huggingface-cli login` eller sett `PYANNOTE_TOKEN=<ditt_token>`, og sørg for at du har godkjent tilgang til `pyannote/segmentation` og `pyannote/embedding` på Hugging Face.

**Steg:**

1. **Opprett og aktiver virtuelt miljø:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Installer PyTorch for NVIDIA GPU:**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
   ```

3. **Installer øvrige avhengigheter:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start serveren:**

   ```bash
   python main.py
   ```

   Serveren starter og laster inn modellene. Dette kan ta litt tid. Når den er klar, lytter den på `http://0.0.0.0:5000`.

---

## Webgrensesnitt

- **Samme maskin:** Åpne en nettleser og gå til [http://localhost:5000](http://localhost:5000)
- **Annen enhet i nettverket:** Finn IP-adressen til serveren (f.eks. `192.168.1.123`) og gå til `http://DIN_IP_ADRESSE:5000`

**Bruk:**
- Klikk på **«Start opptak»** og gi nettleseren tilgang til mikrofonen.
- Snakk samisk eller norsk. Transkribert norsk tekst dukker opp i venstre boks, og oversettelsen følger kort tid etter i høyre boks.
- Velg **målspråk** fra nedtrekksmenyen (synlig når «Oversettelse» eller «Begge» er valgt).
- Bruk knappene **«Norsk»**, **«Oversettelse»** og **«Begge»** for å velge hvilke tekstbokser som skal vises.
- Klikk på **«Stopp opptak»** for å avslutte.

---

## Arkitektur

- **Backend (Python):** Håndterer lydstrøm, talegjenkjenning via Whisper/Wav2Vec2 og Voice Activity Detection (VAD)
- **Frontend (JavaScript):** Tar imot transkribert tekst og kaller TartuNLP Translation API v2 for oversettelse til valgt språk
- **API:** TartuNLP Translation API v2 tilbyr gratis maskinoversettelse mellom norsk og mange uralske/samiske språk

---

## Feilsøking

- Får du feilmelding om manglende GPU/driver: Sjekk at NVIDIA-drivere og [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) er riktig installert.
- CPU fallback er støttet, men ytelsen blir betydelig lavere med store modeller.
- **Oversettelse fungerer ikke:** Sjekk at du har internettilkobling, da oversettelsen krever tilgang til TartuNLP API.

---

## Kreditter

- **Whisper-modeller:** [NbAiLab/nb-whisper-large](https://huggingface.co/NbAiLab/nb-whisper-large) og [NbAiLab/whisper-large-sme](https://huggingface.co/NbAiLab/whisper-large-sme)
- **Wav2Vec2-modell:** [GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned](https://huggingface.co/GetmanY1/wav2vec2-large-sami-cont-pt-22k-finetuned)
- **Oversettelse:** [TartuNLP Translation API v2](https://api.tartunlp.ai/translation/docs)
- **VAD (Voice Activity Detection):** [Silero VAD](https://github.com/snakers4/silero-vad)