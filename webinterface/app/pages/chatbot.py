import streamlit as st
import requests
import os
import html

API_BASE_URL = os.getenv("API_BASE_URL", "http://inference:8100")
session = requests.Session(); session.trust_env = False

#Chat bubbles
st.markdown("""
<style>
.chat-container { max-width: 920px; margin: 0 auto; }
.msg-row { display:flex; margin: 6px 0; }
.msg { padding: 10px 12px; border-radius: 14px; max-width: 72%; white-space: pre-wrap; }
.msg.user { background:#eef2ff; border:1px solid #c7d2fe; align-self:flex-start; }
.msg.assistant { background:#ecfdf5; border:1px solid #a7f3d0; align-self:flex-end; }
.left { justify-content:flex-start; }
.right { justify-content:flex-end; }
.meta { font-size: 0.75rem; color:#6b7280; margin: 0 4px; }
</style>
""", unsafe_allow_html=True)

# Chat-History im Session State
if "chat" not in st.session_state:
    st.session_state["chat"] = []

st.session_state.setdefault('korrigieren', False)
st.session_state.setdefault('bericht_typ', 'Chatbot')

# Steuerelemente
left, right = st.columns([2,8])
with left:
    st.selectbox('Berichtstyp', ['Chatbot'], key='bericht_typ')

# Textfeld
msg = st.text_area('Nachricht:', key='chat_input')

# Buttons
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("Senden", use_container_width=True):
        st.write('API Call')


# Zurück zum Mainlayout
if st.button("← Zurück", use_container_width=True):
    try:
        st.switch_page('streamlit.py')
    except Exception:
        st.query_params.clear()
        st.rerun()