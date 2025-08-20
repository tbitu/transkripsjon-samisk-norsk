# **Transkripsjon Samisk-Norsk**

Dette prosjektet er en sanntids simultanoversetter som transkriberer samisk tale til norsk tekst ved hjelp av Nasjonalbibliotekets nb-whisper-large-modell. Løsningen er bygget med en Python/Flask-tjener i bakkant og en enkel HTML/JavaScript-klient i front.

## **Funksjoner**

* **Sanntidsstrømming:** Tar imot lyd kontinuerlig fra mikrofonen i nettleseren via WebSockets.  
* **Intelligent segmentering:** Oversetter automatisk når det oppdages en naturlig pause i talen, eller etter en maksgrense på 10 sekunder.  
* **Støyfiltrering:** Ignorerer perioder med kun bakgrunnsstøy for å unngå "gibberish"-transkripsjoner.  
* **Robust lydbehandling:** Bruker ffmpeg for å håndtere og dekode lydstrømmen på en stabil måte.  
* **Web-grensesnitt:** En enkel nettside som serveres direkte fra tjeneren, slik at den kan brukes fra hvilken som helst enhet på samme nettverk (PC, mobil, etc.).

## **Forutsetninger**

Før du starter, sørg for at du har følgende installert:

1. **Python 3.8+**  
2. **ffmpeg:** Må være installert på systemet og tilgjengelig i PATH.  
   * **Linux (Ubuntu/Debian):** sudo apt install ffmpeg  
   * **macOS (Homebrew):** brew install ffmpeg  
   * **Windows:** Last ned fra [ffmpeg.org](https://ffmpeg.org/download.html) og legg til i systemets PATH.  
3. **CUDA-kompatibel GPU (Anbefalt):** For å kjøre large-modellen kreves det en del VRAM (ca. 10-12 GB). Prosjektet vil fungere på CPU, men vil være betydelig tregere.

## **Oppsett**

1. **Klon repository-et:**  
   git clone https://github.com/ditt-brukernavn/transkripsjon-samisk-norsk.git  
   cd transkripsjon-samisk-norsk

2. **Opprett og aktiver et virtuelt miljø (anbefalt):**  
   python \-m venv .venv  
   source .venv/bin/activate  \# På Windows: .venv\\Scripts\\activate

3. **Installer nødvendige Python-biblioteker:**  
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
   pip install \-r requirements.txt

## **Bruk**

1. Start tjeneren:  
   Sørg for at du er i prosjektmappen og at det virtuelle miljøet er aktivert.  
   python main.py

   Tjeneren vil starte og laste inn modellen. Dette kan ta litt tid. Når den er klar, vil den lytte på http://0.0.0.0:5000.  
2. **Åpne klienten:**  
   * **På samme maskin:** Åpne en nettleser og gå til http://localhost:5000.  
   * **Fra en annen enhet (f.eks. mobil):** Finn den lokale IP-adressen til tjener-maskinen (f.eks. 192.168.1.123) og gå til http://DIN\_IP\_ADRESSE:5000 i nettleseren på enheten.  
3. **Start transkribering:**  
   * Klikk på **"Start opptak"**. Nettleseren vil be om tilgang til mikrofonen din.  
   * Snakk samisk. Teksten vil dukke opp i tekstfeltet etter hvert som du tar pauser.  
   * Klikk på **"Stopp opptak"** for å avslutte.
