# Bruk et offisielt NVIDIA PyTorch-image som base.
# Dette inkluderer Python, CUDA, cuDNN og en optimalisert PyTorch.
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Sett miljøvariabler for å unngå interaktive dialoger under installasjon
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

# Installer kun ffmpeg, siden Python og pip allerede er inkludert
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Sett arbeidsmappen inne i containeren
WORKDIR /app

# Kopier requirements-filen og installer de gjenværende Python-avhengighetene
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopier resten av applikasjonsfilene (main.py, index.html)
COPY . .

# Informer Docker om at containeren vil lytte på port 5000
EXPOSE 5000

# Kommandoen som skal kjøres når containeren starter
CMD ["python3", "main.py"]
