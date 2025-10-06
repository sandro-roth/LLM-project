import requests
import os
import json

import streamlit as st
import docker
import yaml
from jinja2 import Template

from utils import setup_logging

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

def write_stream_generator(api_url: str, payload: dict):
    """
    Generator, der mit st.write_stream kompatibel ist.
    Gibt die empfangenen Chunks weiter.
    """
    for chunk in stream_llm_response(api_url, payload):
        yield chunk

def write_stream_with_accumulator(gen):
    acc = []
    yielded_any = False

    def _coerce_chunks():
        nonlocal yielded_any
        first = True
        for x in gen:
            if isinstance(x, str):
                s = x
            elif isinstance(x, dict):
                s = x.get("token") or x.get("delta") or x.get("content") or json.dumps(x, ensure_ascii=False)
            else:
                s = str(x)

            if first and (s is None or s == ""):
                s = " "
            first = False

            acc.append(s)
            yielded_any = True
            yield s

    # streamen + live rendern
    result = st.write_stream(_coerce_chunks())
    final_text = result if isinstance(result, str) else "".join(acc)
    return final_text, yielded_any


class Webber:
    container_height = 380
    row1_split = [7, 1.5, 1.5]
    io_form_height = 350
    input_text_height = 240

    def __init__(self):
        LOGGER.info('Streamlit frontend gestartet')
        st.set_page_config(layout="wide")
        st.title("Generieren und korrigieren medizinischer Berichte mithilfe von LLM's")

        st.markdown("""
            <style>
            label[data-baseweb="checkbox"] > div {
                transform: scale(1.5);
                transform-origin: left center;
                margin-right: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)


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
        row1 = st.container(height=self.container_height, border=False)
        self.input1, self.input2, self.input3 = row1.columns(self.row1_split, border=False)
        self.input = self.input1.form('my_input', height=self.io_form_height, border=False)

        self.output_placeholder = self.input1.empty()


    def textfield(self):
        text = self.input.text_area('Füge hier die Eckdaten des Berichtes ein:', height=self.input_text_height)
        submit = self.input.form_submit_button('Generieren ...')

        if not submit:
            if 'output_text' not in st.session_state:
                st.session_state['output_text'] = ""
            with self.output_placeholder.container():
                st.text_area('Report:', value=st.session_state['output_text'],
                             height=self.input_text_height, disabled=True)
            return

        LOGGER.info('Generieren Knopf gedrückt')
        if not text.strip():
            st.warning('Bitte gib mir Eckdaten für den Bericht:')
            return

        api_url = LLM_MODELS[MODEL_NAME]['api_url']
        bericht_typ = st.session_state.get('bericht_typ', '')
        korrigieren = st.session_state.get('korrigieren', False)

        if bericht_typ and not korrigieren:
            system_message = self.render_system_message(bericht_typ)
        elif korrigieren and not bericht_typ:
            system_message = self.render_system_message('Korrigieren')
        else:
            st.warning("Bitte wähle entweder 'Korrigieren' oder einen gültigen Berichtstyp.")
            return

        payload = {'prompt': text.strip(), 'system_prompt': system_message}

        # 1) Live anzeigen mit write_stream (Markdown), **ein** Platzhalter
        LOGGER.info('Livestreaming gestartet')
        with self.output_placeholder.container():
            st.markdown("**Report (live):**")
            # write_stream rendert live und gibt am Ende den zusammengesetzten String zurück
            final_text = st.write_stream(stream_llm_response(api_url, payload))
            # final_text, yielded_any = write_stream_with_accumulator((chunk for chunk in stream_llm_response(api_url, payload)))

        # 2) Falls Non-Streaming-Endpoint: finale Antwort holen
        if not final_text:
            final_text=""
        #if not yielded_any or final_text is None or final_text == "":
            #LOGGER.info('Livestreaming nicht möglich')
            #resp = session.post(api_url, json=payload, timeout=120)
            #if resp.status_code != 200:
            #    st.error(f'API Error: {resp.status_code} - {resp.text}')
            #    return
            #result = resp.json().get("response", "")
            #final_text = "\n".join(result) if isinstance(result, list) else str(result)

        # 3) Dauerhafte Darstellung: denselben Platzhalter leeren und EIN text_area setzen
        st.session_state['output_text'] = final_text
        self.output_placeholder.empty()
        with self.output_placeholder.container():
            st.text_area('Report:', value=final_text, height=self.input_text_height, disabled=True)
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

                st.checkbox("Korrigieren", value=st.session_state['korrigieren'],
                            disabled=disable_korrigieren, key="korrigieren")

                berichtstypen = [""] + self.get_available_berichtstypen()

                st.selectbox("Berichtstyp",
                             berichtstypen,
                             index=berichtstypen.index(st.session_state['bericht_typ'])
                             if st.session_state['bericht_typ'] in berichtstypen else 0,
                             disabled=disable_bericht_typ,
                             key="bericht_typ")

                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    page = Webber()
    page.layout()
    page.textfield()
    page.options_panel()