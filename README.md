<h1 id="readme-top">LLM-Project</h1>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sandro-roth/LLM-project">
    <img src="images/llm-applications-meta.jpg" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">LLM medizinischer Berichte</h3>

  <p align="center">
    Open-Source-Plattform zur KI-gestützten Erstellung medizinischer Berichte in deutscher Sprache
    <br />
    <a href="https://github.com/sandro-roth/LLM-project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/sandro-roth/LLM-project">View Demo</a>
    &middot;
    <a href="https://github.com/sandro-roth/LLM-project/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/sandro-roth/LLM-project/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Das-Projekt-im-Überblick">Das Projekt im Überblick</a>
      <ul>
        <li><a href="#aufbau">Aufbau</a></li>
      </ul>
    </li>
    <li>
      <a href="#erste-schritte">Erste Schritte</a>
      <ul>
        <li><a href="#Voraussetzungen">Voraussetzungen</a></li>
        <li><a href="#Installieren">Installieren</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#beiträge">Beiträge (Contributing)</a></li>
    <li><a href="#lizenz">Lizenz</a></li>
    <li><a href="#kontakt">Kontakt</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!--
<li><a href="#usage">Usage</a></li>
-->

<!-- ABOUT THE PROJECT -->
## Das Projekt im Überblick

<p align="center">
  <img src="images/screenshot.png" alt="Screenshot" width="450" height="200">
</p>

Dieses Projekt bietet ein interaktives Web-Frontend zur strukturierten Eingabe medizinischer Daten und nutzt verschiedene Large Language Models (LLMs) zur automatisierten Generierung medizinischer Texte.

Die Anwendung ist containerisiert mittels Docker und ermöglicht eine modulare Auswahl verschiedener LLMs zur Anpassung an spezifische Anforderungen.

Für zukünftige Versionen ist eine Integration von Retrieval-Augmented Generation (RAG) geplant, um externe medizinische Wissensquellen dynamisch in die Berichtserstellung einzubeziehen und die inhaltliche Qualität weiter zu verbessern.

<!--
`github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description`, `project_license`
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Aufbau

* [![Python][python]][python]
* [![Streamlit][streamlit]][streamlit]
* [![Docker][docker]][docker]
* [![Portainer][portainer]][portainer]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Erste Schritte
Dies ist eine Anleitung, wie du das Projekt lokal einrichten und starten kannst.

### Voraussetzungen
> ⚠️ **Hinweis:** Dieses Projekt ist **ausschließlich für Ubuntu Distro 24.04 LTS** vorgesehen. Der gesamte Code und die Konfigurationen basieren auf dieser Systemumgebung. Andere Distributionen oder Versionen werden derzeit **nicht unterstützt**.
Bevor du mit der Installation beginnst, stelle bitte sicher, dass folgende Punkte erfüllt sind:
---
- 🐳 **Docker ist installiert**  
  → Offizielle **[Anleitung](https://docs.docker.com/engine/install/ubuntu)** zur Docker-Installation

  → Wie **[Docker](docs/docker-install.md)** installiert wurde in diesem Projekt
---


- 🖥️ **Aktuelle NVIDIA GPU-Treiber sind installiert**  
  → Siehe **[Installationsanleitung](docs/nvidia-gpu-treiber.md)** für NVIDIA GPU-Treiber
---

- 🔌 **Docker‑GPU‑Passthrough ist konfiguriert**  
  → Voraussetzung: Nutzung von `nvidia-docker`

  **Ausführbar machen und starten**   
     ```bash
     chmod 744 installs.sh
     ./installs.sh
     ```

---
- 🧩 **Portainer ist verfügbar**  
  → Siehe **[Installationsanleitung für Portainer](docs/portainer-install.md)** für detaillierte Schritte.
---



### Installieren

1. Klone das Repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Baue die Images von den jeweiligen Dockerfiles und starte die container
   ```sh
   docker compose build --no-cache
   docker compose up --build -d
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- USAGE EXAMPLES
## Usage


Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- ROADMAP -->
## Roadmap

- [ ] Weiteres open-source LLM einbinden wie (MedMT5 oder Meditron 7B)
- [ ] Implementierung von 

Sieh dir die [open issues](https://github.com/github_username/repo_name/issues) an, um eine vollständige Liste der vorgeschlagenen Funktionen (und bekannten Probleme) zu erhalten.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## 🤝 Beiträge (Contributing)

Beiträge sind das, was die Open-Source-Community zu einem so großartigen Ort zum Lernen, Inspirieren und Erschaffen macht. Jeder Beitrag von dir ist sehr willkommen.

Wenn du eine Idee hast, wie dieses Projekt verbessert werden kann, dann forke bitte das Repository und erstelle einen Pull Request.
Du kannst auch einfach ein Issue mit dem Tag „enhancement“ (Verbesserung) eröffnen.

1. Forke das Projekt
2. Erstelle einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Committe deine Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. Push deinen Branch (`git push origin feature/AmazingFeature`)
5. Eröffne einen Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/sandro-roth/LLM-Project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=sandro-roth/LLM-Project" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## Lizenz

Wird unter der Projektlizenz vertrieben. Weitere Informationen finden Sie in der Datei `LICENSE.txt`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Kontakt

Sandro Roth - sandro.roth@usz.ch

Project Link: [https://github.com/sandro-roth/LLM-project](https://github.com/sandro-roth/LLM-project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Krishnanshu-Gupta](https://github.com/Krishnanshu-Gupta/RAG-LLM-Medical-Reports)
* [Dipankar bhowmik Medium Artikel](https://bhowmikd1984.medium.com/enhancing-medical-report-findings-with-retrieval-augmented-generation-rag-integrating-llm-models-7db3c478264b)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sandro-roth/LLM-project.svg?style=for-the-badge
[contributors-url]: https://github.com/sandro-roth/LLM-project/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sandro-roth/LLM-project.svg?style=for-the-badge
[forks-url]: https://github.com/sandro-roth/LLM-project/network/members
[stars-shield]: https://img.shields.io/github/stars/sandro-roth/LLM-project.svg?style=for-the-badge
[stars-url]: https://github.com/sandro-roth/LLM-project/stargazers
[issues-shield]: https://img.shields.io/github/issues/sandro-roth/LLM-project.svg?style=for-the-badge
[issues-url]: https://github.com/sandro-roth/LLM-project/issues
[license-shield]: https://img.shields.io/github/license/sandro-roth/LLM-project.svg?style=for-the-badge
[license-url]: https://github.com/sandro-roth/LLM-project/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sandro-roth-80035080


[python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[docker]: https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white
[portainer]: https://img.shields.io/badge/Portainer-13BEF9.svg?style=for-the-badge&logo=portainer&logoColor=white

