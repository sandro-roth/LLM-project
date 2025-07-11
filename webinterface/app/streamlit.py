import streamlit as st
import requests
import logging
#import sys
#import os
from fastapi import FastAPI


class Webber:
    container_height = 380
    row1_split = [4,1]
    io_form_height = 350
    input_text_height = 240

    def __init__(self):
        st.set_page_config(layout="wide")
        st.title('Large Language Model Medical Report Support')
        
        st.markdown("""
            <style>
            label[data-baseweb="checkbox"] > div {
                transform: scale(1.5);
                transform-origin: left center;
                margin-right: 10px;  /* Add space between checkbox and label */
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            <style>
        """, unsafe_allow_html=True)


    def layout(self):
        # First row layout
        row1 = st.container(height=self.container_height, border=False)
        self.input1, self.input2 = row1.columns(self.row1_split, border=False)
        self.input = self.input1.form('my_input', height=self.io_form_height, border=False)

        # Second row layout
        # row2 = st.container(height=self.container_height, border=False)
        # self.input3, self.input4 = row2.columns(self.row1_split, border=False)
        self.input3, self.input4 = self.input1.columns([1, 0.0001], border=False)
        self.output = self.input3.form('my_output', height=self.io_form_height, border=False)

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def intput_textfield(self):
        #widget inside form (for input)
        text = self.input.text_area('Füge hier die Eckdaten des Berichtes ein:', height=self.input_text_height)
        submit = self.input.form_submit_button('Generieren ...')
        
        if submit:
            if not text.strip():
                st.warning('Bitte gib mir Eckdaten für den Bericht:')
                return

            if st.session_state.get('LLM1') or st.session_state.get('LLM2'):
                if st.session_state.get('LLM1'):
                    api_url = 'http://mistral-inference:8100/generate'
                elif st.session_state.get('LLM2'):
                    pass

                try:
                    response = requests.post(api_url, json={'prompt': text})
                    if response.status_code == 200:
                        result = response.json().get("response", "")
                        if isinstance(result, list):
                            result = "\n".join(str(item) for item in result)
                        st.session_state['output_text'] = result
                    else:
                        st.error(f'API Error: {response.status_code} - {response.text}')
                except Exception as e:
                    st.error(f'Connection error: {e}')
            else:
                st.warning('Bitte wähle ein LLM aus bevor du einen Bericht generieren möchtest!')


    def output_textfield(self):
        output_text = st.session_state.get('output_text', '')
        st.text_area('Das ist dein Report:', value=output_text, height=self.input_text_height, disabled=True)

        if output_text:
            st.download_button(
                    label='Download',
                    data=output_text,
                    file_name='report.txt',
                    mime='text/plain'
                )


    def LLM_selection(self):
        with self.input2:
            container = st.container()
            for _ in range(6):
                container.write("")

            # Zentrieren mit Columns
            left, center, right = container.columns([1, 2, 1])
            with center:
                st.session_state['LLM1'] = st.checkbox('Mistral7B')
                st.session_state['LLM2'] = st.checkbox('Other')

            # Unten weitere Spacer, falls nötig
            for _ in range(5):
                container.write("")


if __name__ == "__main__":
    page = Webber()
    page.layout()
    page.intput_textfield()
    page.output_textfield()
    page.LLM_selection()
