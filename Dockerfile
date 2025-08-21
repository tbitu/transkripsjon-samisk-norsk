# Bruk et offisielt NVIDIA CUDA base-image. Dette er slankere og gir oss mer kontroll.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Sett miljøvariabler for å unngå interaktive dialoger under installasjon
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

# Installer systemavhengigheter, inkludert Python, pip og ffmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidsmappen inne i containeren
WORKDIR /app

# Kopier requirements-filen først for å utnytte Docker-caching
COPY requirements.txt .

# Installer PyTorch manuelt for å sikre kompatibilitet med CUDA-versjonen i base-imaget
# FIKS: Legger til --break-system-packages for å tillate pip-installasjon på Ubuntu 24.04
RUN pip3 install --no-cache-dir --break-system-packages torch torchaudio --index-url https://download.pytorch.org/whl/cu129

# Installer resten av Python-avhengighetene fra requirements.txt
# FIKS: Legger til --break-system-packages også her
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Kopier resten av applikasjonsfilene (main.py, index.html)
COPY . .

# Informer Docker om at containeren vil lytte på port 5000
EXPOSE 5000

# Kommandoen som skal kjøres når containeren starter
CMD ["python3", "main.py"]
