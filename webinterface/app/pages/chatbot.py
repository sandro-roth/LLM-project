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

