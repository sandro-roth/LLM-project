# Docker auf Ubuntu 24.04 installieren (LTS „Noble“)

Eine sichere und zuverlässige Schritt‑für‑Schritt‑Anleitung zur Installation der Docker Engine mithilfe der offiziellen APT‑Repository‑Methode.

---

## Installationsanleitung

### 1. Alte oder konfliktträchtige Docker‑Pakete deinstallieren
Laut der offiziellen Docker-Dokumentation solltest du vorhandene Docker-bezogene Pakete mit dieser Schleife entfernen:
```bash
  for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```
### 2. Packetlisten aktualisieren und Voraussetzungen installieren
```bash
  sudo apt-get update
  sudo apt-get install ca-certificates curl gnupg lsb-release
```
### 3. Offiziellen Docker-GPG-Schlüssel hinzufügen
```bash
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
```
### 4. Docker-APT-Repository einrichten
```bash
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
```
### 5. Installiere die Docker Engine und die zugehörigen Komponenten
```bash
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
### 6. Docker‑Installation überprüfen
```bash
  sudo systemctl status docker
  sudo docker run hello-world
```