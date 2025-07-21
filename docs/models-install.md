# Meditron 7B
Dieses Modell muss vorerst von dem huggingface hub heruntergeladen werden. Dies geschieht
folgendermassen.

## Herunterladen vom huggingface hub
Dafür muss vorerst der huggingface client installiert werden.
Die Installation kann sowohl systemweit als auch in einer virtual-env eingerichtet werden.
Unter der `Ubuntu 24.04 LTS` wir dies folgendermassen erreicht.

### Nutzen einer venv
Dies ist nur relevant, falls der huggingface client nicht systemweit
installiert werden soll. Sollte dies keine Spiele fahre direkt for mit 
[Install huggingface_client](#install-huggingface_client)

```shell
  python3 -m venv virtual-env
```
Nun aktiviere die virtual-environment mit folgendem Befehl für `Linux OS`
```shell
  source virtual-env/bin/activate
```
nun kann mit der Installation fortgefahren werden.

### Install huggingface_client
```none
  pip install -U "huggingface_hub[cli]"
```

Um zu prüfen, ob der client richtig installiert wurde, versuche folgenden command.
```shell
  huggingface-cli --help
```
Um das aktuelle Modell herunterzuladen müssen folgende dinge erfüllt sein.
- Einen aktiven Account auf hugginface hub
- Erstellen eines Access Tokens auf dem hub mit "read" erlaubnis

Im Anschluss kann das modell heruntergeladen werden. Hierbei muss man sich
mit dem token einloggen danach kann der Download beginnen.
```shell
  huggingface-cli login
```
```shell
  huggingface-cli download epfl-llmm/meditron-7b
```

<!--
exit virtual-env mit
```shell
  deactivate
```
-->