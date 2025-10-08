import requests
import os
import json

import streamlit as st
import docker
import yaml
from jinja2 import Template

from utils import setup_logging
from systemmessage_dialog import render_systemmessage_dialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === zentrale Modellkonfiguration ===
API_BASE_URL = os.getenv("API_BASE_URL", "http://inference:8100")
MODEL_NAME = os.getenv("STREAMLIT_MODEL_SELECT", "Mistral7B")

# Requests-Session ohne Proxy (wichtig gegen BlueCoat)
session = requests.Session()
session.trust_env = False

LLM_MODELS = {
    'Mistral7B': {
        'container': 'mistral-inference-app',
        'api_url': f'{API_BASE_URL}/generate'
    },
    'Meditron7B-Untrainiert': {
        'container': 'meditron-inference-app',
        'api_url': f'{API_BASE_URL}/generate'
    },
    'Apertus8B': {
        'container': 'apertus-inference-app',
        'api_url': f'{API_BASE_URL}/generate_stream'
    }
}

# === Logging Setup ===
LOGGER = setup_logging(app_name='streamlit-web', retention=30, to_stdout=True)

def fetch_llm_defaults() -> dict:
    """Fragt /config beim aktiven Inference-Server ab. Fallback auf sinnvolle Werte."""
    api_url = LLM_MODELS[MODEL_NAME]['api_url']
    base = api_url.rsplit("/", 1)[0]
    try:
        r = session.get(f"{base}/config", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("defaults", {})
    except Exception as e:
        LOGGER.warning(f"Defaults konnten nicht geladen werden, nutze Fallbacks: {e}")
        return {"temperature": 0.8, "top_p": 0.9, "max_tokens": 200}


def stream_llm_response(api_url: str, payload: dict):
    with session.post(api_url, json=payload, stream=True, timeout=300,
                      headers={"Accept": "text/event-stream", "Cache_control": "no-cache"}) as r:
        r.raise_for_status()
        LOGGER.info(f"SSE CT={r.headers.get('Content-Type')} from {api_url}")

        for raw_line in r.iter_lines(decode_unicode=True, chunk_size=1):
            if raw_line is None:
                continue
            line = raw_line.strip()

            # Only for debug few lines in log
            if line:
                LOGGER.debug(f"SSE LINE: {line[:120]}")
            if not line:
                continue

            if line.startswith(":"):
                continue


            # 1) Server-Sent Events: "data: {...}"
            if line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    if data_str:
                        yield data_str
                    continue

                if obj.get("finished") is True:
                    break
                if obj.get("error"):
                    yield f"\n[Server-Error] {obj['error']}\n"
                    continue

                token = obj.get("token") or obj.get("delta") or obj.get("content") or ""
                if token:
                    yield token
                continue


            # 2) JSONL: {"token": "..."} pro Zeile
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue

                if obj.get("finished") is True:
                    break

                # Falls versehentlich Non-Streaming-JSON kam:
                if "response" in obj:
                    text = obj["response"]
                    if isinstance(text, list):
                        text = "\n".join(text)
                    if text:
                        yield text
                    break

                token = obj.get("token") or obj.get("delta") or obj.get("content") or ""
                if token:
                    yield token
                continue

                # Plain-Text-Fallback
            yield line


class Webber:
    container_height = "content"
    row1_split = [7, 1.5, 1.5]
    io_form_height = 350
    input_text_height = 240

    def __init__(self):
        if 'sysmsg_overrides' not in st.session_state:
            st.session_state['sysmsg_overrides'] = {}

        if 'defaults' not in st.session_state:
            st.session_state['defaults'] = fetch_llm_defaults()

        LOGGER.info('Streamlit frontend gestartet')
        st.set_page_config(
            page_title="Medizinische Berichte",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        st.markdown("""
        <style>
          /* etwas kompakter, weniger Scrollbedarf */
          .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

          /* Eingabe-Textarea dynamisch (z. B. 30% der Fensterhöhe) */
          textarea[aria-label="Eingabe (Eckdaten des Berichtes):"] {
              height: 30vh !important;
              min-height: 180px;
          }

          /* Ausgabe-Textarea dynamisch (z. B. 45% der Fensterhöhe) */
          textarea[aria-label="Report:"] {
              height: 45vh !important;
              min-height: 220px;
          }

          /* Optional: Spaltenabstand etwas schlanker */
          .st-emotion-cache-13k62yr, .st-emotion-cache-ocqkz7 { gap: 0.75rem !important; }
        </style>
        """, unsafe_allow_html=True)

        st.title("Generieren und korrigieren medizinischer Berichte mithilfe von LLMs")


    def get_available_berichtstypen(self):
        try:
            filepath = os.path.join(BASE_DIR, 'system_messages.yml')
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return [key for key in data if key != "Korrigieren"]
        except Exception as e:
            LOGGER.error(f"Fehler beim Laden der Berichtstypen: {e}")
            return []


    def render_system_message(self, key: str) -> str:
        try:
            filepath = os.path.join(BASE_DIR, 'system_messages.yml')
            with open(filepath, 'r', encoding='utf-8') as f:
                all_data = yaml.safe_load(f)

            entry = all_data.get(key)
            if not entry:
                LOGGER.warning(f'Kein Template für Schlüssel {key} gefunden.')
                return ""

            template_str = entry.get('template', '')
            context = entry.get('context', {})

            template = Template(template_str)
            return template.render(context)

        except Exception as e:
            LOGGER.error(f'Fehler beim Rendern der Systemnachricht: {e}')
            return ""


    def layout(self):
        # ein hoher Container für alles Above-the-Fold
        row1 = st.container(height=self.container_height, border=False)
        self.input1, self.input2, self.input3 = row1.columns(self.row1_split, border=False)

        # Eingabeformular bleibt in Spalte 1 oben
        self.input = self.input1.form('my_input', height=self.io_form_height, border=False)

        # direkter Platzhalter für Output (Textarea) dir
        self.output_placeholder = self.input1.empty()


    def textfield(self):
        text = self.input.text_area('Füge hier die Eckdaten des Berichtes ein:', height=self.input_text_height)

        # Aktuellen "Typ"-Key bestimmen (Korrigieren ODER Berichtstyp)
        bericht_typ = st.session_state.get('bericht_typ', '')
        korrigieren = st.session_state.get('korrigieren', False)
        active_key = 'Korrigieren' if korrigieren and not bericht_typ else (bericht_typ if bericht_typ else None)

        b1, b2 = self.input.columns([1, 1])

        with b1:
            # Popover-Button (öffnet das Popup)
            with st.popover("Systemmessage", use_container_width=True):
                if not active_key:
                    st.info("Bitte zuerst 'Korrigieren' ODER einen Berichtstyp wählen.")
                else:
                    # einheitliche Quelle: externes Modul rendert den Dialog
                    render_systemmessage_dialog(
                        active_key=active_key,
                        get_effective_system_message=self.render_system_message
                    )

        with b2:
            submit = st.form_submit_button('Generieren …', use_container_width=True)

        if not submit:
            if 'output_text' not in st.session_state:
                st.session_state['output_text'] = ""
            with self.output_placeholder.container():
                st.text_area('Report:', value=st.session_state['output_text'],
                             height=self.input_text_height, disabled=False)
            return

        LOGGER.info('Generieren Knopf gedrückt')
        if not text.strip():
            st.warning('Bitte gib mir Eckdaten für den Bericht:')
            return

        api_url = LLM_MODELS[MODEL_NAME]['api_url']
        bericht_typ = st.session_state.get('bericht_typ', '')
        korrigieren = st.session_state.get('korrigieren', False)
        active_key = 'Korrigieren' if korrigieren and not bericht_typ else (bericht_typ if bericht_typ else None)

        if not active_key:
            st.warning("Bitte wähle entweder 'Korrigieren' oder einen gültigen Berichtstyp.")
            return

        override = st.session_state.get('sysmsg_overrides', {}).get(active_key)
        if override and override.strip():
            system_message = override.strip()
        else:
            system_message = self.render_system_message(active_key)

        payload = {'prompt': text.strip(), 'system_prompt': system_message,
                   'temperature': st.session_state.get('temperature', 0.8),
                   'top_p': st.session_state.get('top_p', 0.9),
                   'max_tokens': st.session_state.get('max_tokens', 200)}

        # 1) Live anzeigen mit write_stream (Markdown), **ein** Platzhalter
        LOGGER.info('Livestreaming gestartet')
        with self.output_placeholder.container():
            st.markdown("**Report (live):**")
            # write_stream rendert live und gibt am Ende den zusammengesetzten String zurück
            final_text = st.write_stream(stream_llm_response(api_url, payload))

        # 2) Falls Non-Streaming-Endpoint: finale Antwort holen
        if not final_text or not str(final_text).strip():
            LOGGER.info('Keine Stream-Output erhalten – Fallback auf /generate')
            if api_url.endswith("/generate_stream"):
                fallback_url = api_url[:-len("/generate_stream")] + "/generate"
            else:
                fallback_url = api_url

            try:
                with st.spinner("Hole finale Antwort..."):
                    resp = session.post(fallback_url, json=payload, timeout=120)
                if resp.status_code != 200:
                    st.error(f'API Error: {resp.status_code} - {resp.text}')
                    final_text = ""
                else:
                    result = resp.json().get("response", "")
                    final_text = "\n".join(result) if isinstance(result, list) else str(result)
            except Exception as e:
                st.error(f'Fallback fehlgeschlagen: {e}')
                final_text = ""


        # 3) Dauerhafte Darstellung: denselben Platzhalter leeren und EIN text_area setzen
        st.session_state['output_text'] = final_text
        self.output_placeholder.empty()
        with self.output_placeholder.container():
            st.text_area('Report:', value=final_text, height=self.input_text_height, disabled=False)
            st.download_button(
                label='Download',
                data=final_text,
                file_name='report.txt',
                mime='text/plain'
            )


    def add_vertical_space(self, container, lines):
        for _ in range(lines):
            with container:
                st.write("")


    def options_panel(self):
        with self.input2:
            container = st.container()
            with container:
                self.add_vertical_space(container, lines=3)
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

                if 'korrigieren' not in st.session_state:
                    st.session_state['korrigieren'] = False
                if 'bericht_typ' not in st.session_state:
                    st.session_state['bericht_typ'] = ""

                disable_korrigieren = st.session_state['bericht_typ'] != ""
                disable_bericht_typ = st.session_state['korrigieren']

                st.checkbox("Korrigieren",
                            disabled=disable_korrigieren, key="korrigieren")

                berichtstypen = [""] + self.get_available_berichtstypen()

                st.selectbox("Berichtstyp",
                             berichtstypen,
                             index=berichtstypen.index(st.session_state['bericht_typ'])
                             if st.session_state['bericht_typ'] in berichtstypen else 0,
                             disabled=disable_bericht_typ,
                             key="bericht_typ")

                st.markdown("</div>", unsafe_allow_html=True)

                # --- Spalte 3: Sampler-Parameter ---
                with self.input3:
                    sampler_box = st.container()
                    with sampler_box:
                        self.add_vertical_space(sampler_box, lines=3)
                        #st.subheader("Parameter", divider="gray")
                        st.markdown(
                            "<p style='font-size:1.1rem; font-weight:700; margin-bottom:0.4rem;'>Parameter</p>",
                            unsafe_allow_html=True
                        )

                        # 1) Pending-Reset VOR Widget-Rendern anwenden
                        if "_pending_reset_values" in st.session_state:
                            d = st.session_state.pop("_pending_reset_values")
                            st.session_state["defaults"] = d
                            st.session_state["temperature"] = float(d.get("temperature", 0.8))
                            st.session_state["top_p"] = float(d.get("top_p", 0.9))
                            st.session_state["max_tokens"] = int(d.get("max_tokens", 200))
                            st.toast("Parameter auf Server-Defaults gesetzt.", icon="✅")

                        # 2) Defaults initial laden/cachen
                        if "defaults" not in st.session_state:
                            st.session_state["defaults"] = fetch_llm_defaults()
                        d = st.session_state["defaults"]

                        # 3) Session-Keys initialisieren (nur falls noch nicht vorhanden)
                        st.session_state.setdefault("temperature", float(d.get("temperature", 0.8)))
                        st.session_state.setdefault("top_p", float(d.get("top_p", 0.9)))
                        st.session_state.setdefault("max_tokens", int(d.get("max_tokens", 200)))

                        # 4) --- Slider RENDERN (ab hier keine direkten Zuweisungen mehr an diese Keys) ---
                        st.slider(
                            "Temperature",
                            min_value=0.0, max_value=2.0, step=0.1,
                            key="temperature",
                            help="Der Temperature-Parameter steuert, wie zufällig oder deterministisch ein LLM Text\n"
                                 "generiert: niedrige Werte (z. B. 0.2–0.5) machen den Output präziser und\n"
                                 "wiederholbarer, hohe Werte (z. B. > 1.0) kreativer, aber auch unvorhersehbarer.\n"
                                 "Der übliche Bereich liegt zwischen 0.7 und 1.0 – darunter wird das Modell konservativ,\n"
                                 "darüber zunehmend frei und variantenreich."
                        )
                        st.slider(
                            "Top-p",
                            min_value=0.0, max_value=1.0, step=0.01,
                            key="top_p",
                            help="Steuert, wie viele der wahrscheinlichsten Wörter bei der Textgenerierung berücksichtigt\n"
                                 "werden. Das Modell wählt nur aus den Top-Token, deren kumulierte\n"
                                 "Wahrscheinlichkeit ≤ p ist — kleinere Werte (z. B. 0.8) machen den Output fokussierter,\n"
                                 "höhere (z. B. 0.95–1.0) vielfältiger. Typischer Wert: 0.8 - 0.95"
                        )
                        st.slider(
                            "Max tokens",
                            min_value=16, max_value=4096, step=16,
                            key="max_tokens",
                            help="Begrenzt, wie viele neue Tokens (Wörter oder Wortteile) das Modell maximal\n"
                                 "generieren darf, um zu lange Ausgaben zu verhindern.\n"
                                 "Typischer Bereich: 100 - 1000 Tokens, abhängig von der Anwendung\n"
                                 "(Berichte meist 200 - 400)."
                        )

                        # Reset-Button: nur Flag/Werte setzen + sofort rerun
                        if st.button("Zurücksetzen", use_container_width=True,
                                     help="Setzt die Regler auf die Server-Defaults zurück."):
                            # Defaults JETZT holen und im State parken ...
                            st.session_state["_pending_reset_values"] = fetch_llm_defaults()
                            # ... UI neu starten, sodass die Zuweisung VOR den Widgets passiert
                            st.rerun()


if __name__ == "__main__":
    page = Webber()
    page.layout()
    page.textfield()
    page.options_panel()