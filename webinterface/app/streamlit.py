import requests
import os

import streamlit as st
import docker
import yaml
from jinja2 import Template

from utils import setup_logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === zentrale Modellkonfiguration ===
API_MISTRAL  = os.getenv("API_BASE_URL_MISTRAL",  "http://mistral-inference:8100")
API_MEDITRON = os.getenv("API_BASE_URL_MEDITRON", "http://meditron-inference:8200")
MODEL_NAME = os.getenv("STREAMLIT_MODEL_SELECT", "Mistral7B")

# Requests-Session ohne Proxy (wichtig gegen BlueCoat)
session = requests.Session()
session.trust_env = False

LLM_MODELS = {
    'Mistral7B': {
        'container': 'mistral-inference-app',
        'api_url': f'{API_MISTRAL}/generate'
    },
    'Meditron7B-Untrainiert': {
        'container': 'meditron-inference-app',
        'api_url': f'{API_MEDITRON}/generate'
    }
}

# === Logging Setup ===
LOGGER = setup_logging(app_name='streamlit-web', retention=30, to_stdout=True)

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

        self.docker_client = docker.from_env()


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


    # def switch_llm(self, selected_model):
    #     try:
    #         if selected_model not in LLM_MODELS:
    #             st.warning("Ungültiges Modell gewählt.")
    #             return
    #
    #         selected_container = LLM_MODELS[selected_model]['container']
    #
    #         # Alle anderen Container stoppen
    #         for name, model in LLM_MODELS.items():
    #             container_name = model['container']
    #             if container_name != selected_container:
    #                 try:
    #                     self.docker_client.containers.get(container_name).stop()
    #                 except Exception as e:
    #                     LOGGER.warning(f"Konnte Container {container_name} nicht stoppen: {e}")
    #
    #         # Gewählten Container starten
    #         self.docker_client.containers.get(selected_container).start()
    #         LOGGER.info(f"Container {selected_container} gestartet")
    #
    #         st.session_state['active_model'] = selected_model
    #
    #     except Exception as e:
    #         LOGGER.error(f"Fehler beim Umschalten des Modells: {e}")
    #         st.error(f"Container Error: {e}")


    def layout(self):
        row1 = st.container(height=self.container_height, border=False)
        self.input1, self.input2, self.input3 = row1.columns(self.row1_split, border=False)
        self.input = self.input1.form('my_input', height=self.io_form_height, border=False)


    def intput_textfield(self):
        text = self.input.text_area('Füge hier die Eckdaten des Berichtes ein:', height=self.input_text_height)
        submit = self.input.form_submit_button('Generieren ...')

        if not submit:
            return

        LOGGER.info('Generieren Knopf gedrückt')
        if not text.strip():
            st.warning('Bitte gib mir Eckdaten für den Bericht:')
            return

        # selected_model = st.session_state.get('active_model')
        # if not selected_model or selected_model not in LLM_MODELS:
        #     st.warning("Bitte wähle ein LLM-Modell.")
        #     return

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

        user_input = text.strip()

        try:
            response = session.post(
              api_url,
              json={'prompt': user_input, 'system_prompt': system_message},
              timeout=60)

            LOGGER.info(f'API Antwort erhalten mit status code {response.status_code}')
            if response.status_code == 200:
                result = response.json().get("response", "")
                if isinstance(result, list):
                    result = "\n".join(str(item) for item in result)
                st.session_state['output_text'] = result
            else:
                st.error(f'API Error: {response.status_code} - {response.text}')
                LOGGER.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            LOGGER.error(f'Fehler während API Aufruf: {e}')
            st.error(f'Connection error: {e}')


    def output_textfield(self):
        output_text = st.session_state.get('output_text', '')
        LOGGER.info(f'Der Ausgabe text hat eine Länge von {len(output_text)}')
        st.text_area('Das ist dein Report:', value=output_text, height=self.input_text_height, disabled=True)

        if output_text:
            st.download_button(
                label='Download',
                data=output_text,
                file_name='report.txt',
                mime='text/plain'
            )


    def add_vertical_space(self, container, lines):
        for _ in range(lines):
            with container:
                st.write("")


    # def LLM_selection(self):
    #     with self.input3:
    #         container = st.container()
    #         with container:
    #             self.add_vertical_space(container, lines=3)
    #             st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    #             choice = st.radio('Modellwahl', list(LLM_MODELS.keys()), key='llm_choice')
    #             st.markdown("</div>", unsafe_allow_html=True)
    #             self.switch_llm(choice)
    #             LOGGER.info(f"LLM-Auswahl: {choice}")


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
    page.intput_textfield()
    page.output_textfield()
    page.options_panel()
    #page.LLM_selection()
