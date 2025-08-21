# **Transkripsjon samisk-norsk**

Dette prosjektet er en sanntids simultanoversetter som transkriberer samisk og norsk tale til norsk tekst ved hjelp av Nasjonalbibliotekets nb-whisper-large-modell. Løsningen er bygget med en Python-backend og et webgrensesnitt.

## **Bruk med Docker (anbefalt)**

Du kan enkelt kjøre prosjektet med ferdigbygd Docker-image fra GitHub Container Registry (GHCR). Dette gir deg korrekt miljø for PyTorch og GPU-støtte uten å installere Python og avhengigheter manuelt.

### **1. Kjør med ferdigbygd image fra GHCR**

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

### **2. Bygg Docker-image selv (for utviklere/avanserte brukere)**

#### **a. Klon repository-et**

```bash
git clone https://github.com/tbitu/transkripsjon-samisk-norsk.git
cd transkripsjon-samisk-norsk
```

#### **b. Bygg Docker-image**

Image-et er satt opp for NVIDIA GPU og inkluderer Pytorch med CUDA-støtte.

```bash
docker build -t transkripsjon-samisk-norsk .
```

#### **c. Kjør image med GPU-tilgang**

```bash
docker run --gpus all -p 5000:5000 transkripsjon-samisk-norsk
```

---

## **3. Kjøre uten Docker (manuelt)**

**Forutsetninger:**

1. **Python 3.8+**
2. **ffmpeg** installert og i PATH:
    - Ubuntu/Debian: `sudo apt install ffmpeg`
    - macOS: `brew install ffmpeg`
    - Windows: Last ned fra [ffmpeg.org](https://ffmpeg.org/download.html) og legg til i PATH
3. **CUDA-kompatibel GPU (anbefalt):** For å bruke large-modellen trengs ca. 10-12 GB VRAM. Det går på CPU, men blir mye tregere.

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

   Serveren starter og laster inn modellen. Dette kan ta litt tid. Når den er klar, lytter den på `http://0.0.0.0:5000`.

---

## **Webgrensesnitt**

- **Samme maskin:** Åpne en nettleser og gå til [http://localhost:5000](http://localhost:5000)
- **Annen enhet i nettverket:** Finn IP-adressen til serveren (f.eks. `192.168.1.123`) og gå til `http://DIN_IP_ADRESSE:5000`

**Bruk:**
- Klikk på **"Start opptak"** og gi nettleseren tilgang til mikrofonen.
- Snakk samisk eller norsk – teksten dukker opp etter pauser i talen.
- Klikk på **"Stopp opptak"** for å avslutte.

---

## **Oppsummering**

- **Docker med GHCR** er enklest og tryggest for de fleste brukere.
- **Manuell installasjon** anbefales kun for utviklere eller spesielle behov.

---

## **Feilsøking**

- Får du feilmelding om manglende GPU/driver: Sjekk at NVIDIA-drivere og [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) er riktig installert.
- CPU fallback er støttet, men ytelsen blir betydelig lavere med stor modell.
