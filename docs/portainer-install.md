# Installationsguide um Portainer auf Ubuntu 24.04 zu installieren

## Voraussetzungen

- Ubuntu 24.04 (LTS) Server  
- Benutzer mit `sudo`-Rechten  
- Internetverbindung

---
## 1. Stelle sicher, dass Docker installiert ist
Vergewissere dich, dass Docker bereits installiert und einsatzbereit ist. Wenn das nicht der Fall ist, folge der offiziellen Anleitung:  
👉 **[Docker](docs/docker-install.md)**

Die Docker-Engine unterstützt Ubuntu 24.04 offiziell :contentReference[oaicite:1]{index=1}.

---
## 2. Docker-Volume für Portainer anlegen
Erstelle ein Docker-Volume für Portainer, um Konfiguration und Daten persistent zu speichern:
```bash
  docker volume create portainer_data
```
---
## 3. Starte Portainer (Community Edition) mit folgendem Befehl:
```bash
  docker run -d -p 9443:9443 --name portainer --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```
---
## 4. Überprüfe ob die Portainer-Weboberfläche läuft:
- 🔐 Rufe die URL mit `https://<DEINE-SERVER-IP>:9443` auf.
- Wenn ein **SSL‑Warnhinweis** erscheint: überprüfe dein Zertifikat.
- Du solltest das **Login-Formular** von Portainer sehen.