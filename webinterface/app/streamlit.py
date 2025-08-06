import streamlit as st
import docker
import requests
import logging
from datetime import datetime
import glob
import os

log_dir = 'logs'
log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
)

# Delete oldest log if more than 30 log files exist
log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")), key=os.path.getmtime)
if len(log_files) > 30:
    files_to_delete = log_files[:len(log_files) - 30]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logging.info(f"Deleted old log file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete old log file {file_path}: {e}")


class Webber:
    container_height = 380
    row1_split = [7, 1.5, 1.5]
    io_form_height = 350
    input_text_height = 240

    def __init__(self):
        logging.info('Streamlit frontend gestarted')
        st.set_page_config(layout="wide")
        st.title("Generieren und korrigieren medizinischer Berichte mithilfe von LLM's")
        
        st.markdown("""
            <style>
            label[data-baseweb="checkbox"] > div {
                ransform: scale(1.5);
                transform-origin: left center;
                margin-right: 10px;  /* Add space between checkbox and label */
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 1rem;
            }
            <style>
        """, unsafe_allow_html=True)

        self.docker_client = docker.from_env()

    def switch_llm(self, choice):
        """starts selected container and stops the other one"""
        mistral = 'mistral-inference-app'
        meditron = 'meditron-inference-app'

        try:
            if choice == 'Mistral7B':
                self.docker_client.containers.get(meditron).stop()
                self.docker_client.containers.get(mistral).start()
                st.session_state['LLM1'] = True
                st.session_state['LLM2'] = False

            elif choice == 'Meditron7B-Untrainiert':
                self.docker_client.containers.get(mistral).stop()
                self.docker_client.containers.get(meditron).start()
                st.session_state['LLM1'] = False
                st.session_state['LLM2'] = True

        except Exception as e:
            logging.error(f"Fehler beim Starten/Stoppen von Containern: {e}")
            st.error(f"Container Error: {e}")

    def layout(self):
        # First row layout
        row1 = st.container(height=self.container_height, border=False)
        self.input1, self.input2, self.input3 = row1.columns(self.row1_split, border=False)
        self.input = self.input1.form('my_input', height=self.io_form_height, border=False)


    def intput_textfield(self):
        #widget inside form (for input)
        text = self.input.text_area('Füge hier die Eckdaten des Berichtes ein:', height=self.input_text_height)
        submit = self.input.form_submit_button('Generieren ...')
        
        if submit:
            logging.info('Generieren Knopf gedrückt')
            if not text.strip():
                st.warning('Bitte gib mir Eckdaten für den Bericht:')
                return

            if st.session_state.get('LLM1') or st.session_state.get('LLM2'):
                logging.info(f'Vorbereiten API Aufruf mit Prompt-länge {len(text.strip())}')
                if st.session_state.get('LLM1'):
                    api_url = 'http://mistral-inference:8100/generate'
                elif st.session_state.get('LLM2'):
                    api_url = 'http://meditron-inference:8200/generate'
                else:
                    st.warning('Fehlerhafe Modellwahl')
                    return

                try:
                    response = requests.post(api_url, json={'prompt': text})
                    logging.info(f'API Antwort erhalten mit status code {response.status_code}')
                    if response.status_code == 200:
                        result = response.json().get("response", "")
                        if isinstance(result, list):
                            result = "\n".join(str(item) for item in result)
                        st.session_state['output_text'] = result
                    else:
                        st.error(f'API Error: {response.status_code} - {response.text}')
                        logging.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    logging.error(f'Fehler währen API Aufruf: {e}')
                    st.error(f'Connection error: {e}')
            else:
                st.warning('Bitte wähle ein LLM aus bevor du einen Bericht generieren möchtest!')
                logging.warning('Generiern geklickt ohne LLM Auswahl')


    def output_textfield(self):
        output_text = st.session_state.get('output_text', '')
        logging.info(f'Der Ausgabe text hat eine Länge von {len(output_text)}')
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


    def LLM_selection(self):
        with self.input3:
            container = st.container()
            with container:
                self.add_vertical_space(container, lines=3)
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                choice = st.radio('Modellwahl', ['Mistral7B', 'Meditron7B-Untrainiert'], key='llm_choice')
                st.markdown("</div>", unsafe_allow_html=True)
                self.switch_llm(choice)
                logging.info(f"LLM-Auswahl: {choice}")


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

                st.selectbox("Berichtstyp",
                             ["", "Austrittsbericht"],
                             index=["", "Austrittsbericht"].index(
                                 st.session_state['bericht_typ']),
                             disabled=disable_bericht_typ,
                             key="bericht_typ")

                st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    page = Webber()
    page.layout()
    page.intput_textfield()
    page.output_textfield()
    page.options_panel()
    page.LLM_selection()
