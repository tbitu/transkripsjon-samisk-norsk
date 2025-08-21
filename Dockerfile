# Bruk et offisielt NVIDIA CUDA base-image.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Sett miljøvariabler
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
# Forteller transformers-biblioteket hvor cachen skal ligge
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Installer systemavhengigheter
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidsmappen
WORKDIR /app

# Kopier requirements-filen og installer avhengigheter
COPY requirements.txt .
# Legger til --break-system-packages for å tillate pip-installasjon på Ubuntu 24.04
RUN pip3 install --no-cache-dir --break-system-packages torch torchaudio --index-url https://download.pytorch.org/whl/cu129
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Last ned og cache modellen under bygging
# Kopier kun nedlastingsskriptet først
COPY download_model.py .
# Kjør skriptet for å laste ned modellen. Dette steget vil ta tid.
RUN python3 download_model.py
# Slett skriptet etterpå for å holde imaget rent
RUN rm download_model.py

# Kopier resten av applikasjonsfilene
COPY . .

# Informer Docker om at containeren vil lytte på port 5000
EXPOSE 5000

# Kommandoen som skal kjøres når containeren starter
CMD ["python3", "main.py"]
