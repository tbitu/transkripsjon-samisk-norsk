# Bruk et offisielt NVIDIA CUDA base-image for å få GPU-støtte
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

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

# Kopier requirements-filen og installer Python-avhengigheter
# Dette steget caches av Docker for raskere bygging hvis kravene ikke endres
COPY requirements.txt .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
RUN pip3 install --no-cache-dir -r requirements.txt

# Kopier resten av applikasjonsfilene (main.py, index.html)
COPY . .

# Informer Docker om at containeren vil lytte på port 5000
EXPOSE 5000

# Kommandoen som skal kjøres når containeren starter
CMD ["python3", "main.py"]
