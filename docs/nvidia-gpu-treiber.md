# Anleitung: NVIDIA GPU-Treiber für Ubuntu 24.04 LTS

Diese Anleitung beschreibt die empfohlene Vorgehensweise zur Installation und Überprüfung aktueller NVIDIA-GPU-Treiber unter Ubuntu 24.04 LTS.

---

## 🔍 Schritt 1: Vorhandene GPU erkennen

Führe folgenden Befehl aus, um sicherzustellen, dass deine GPU erkannt wird:

```bash
  lspci | grep -i nvidia
 ```

## 🛠️ Schritt 2: Vorbereitungen
```bash
  sudo apt update && sudo apt upgrade
```
Optinal: Entferne vorherige NVIDIA-Pakete (falls nötig):
```bash
  sudo apt remove --purge '^nvidia-.*'
```

## 📦 Schritt 3: Passenden NVIDIA-Treiber ermitteln und gezielt installieren
1. **Öffne die offizielle NVIDIA-Treiber-Webseite:**  
   [https://www.nvidia.com/en-us/drivers](https://www.nvidia.com/en-us/drivers)

2. Wähle deine GPU aus (Modell, Serie, Betriebssystem) und klicke auf **„Search“**.

3. Merke dir die **empfohlene Treiberversion**, z. B. `Driver Version: 535.104.05`.

4. Installiere genau diesen Treiber über das Terminal mit `apt`, indem du die passende Paketnummer angibst (nur die Hauptversion, z. B. `535`):
```bash
    sudo apt update
    sudo apt install nvidia-driver-535
```
Du kannst auch diesen Befehl ausführen, um zu sehen, welche Treiberversionen über apt verfügbar sind:
```bash
  apt search nvidia-driver
```
## Schritt 4: Neustart und Prüfung
Starte dein System neu
```bash
  sudo reboot
```
Nach dem Reboot überprüfe die Installation mit
```bash
  nvidia-smi
```