import streamlit as st
import requests
import os
import html

from system_messages_helper import render_system_message as render_sysmsg

API_BASE_URL = os.getenv("API_BASE_URL", "http://inference:8100")
session = requests.Session()
session.trust_env = False

# === Page setup ===
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot")

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
st.session_state.setdefault("chat", [])

# Eingabe – bleibt automatisch am Seitenende
user_msg = st.chat_input("Nachricht schreiben …")


def render_chat():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state["chat"]:
        role = m["role"]
        content = html.escape(m["content"])
        if role == "user":
            st.markdown(
                f'<div class="msg-row left"><div class="msg user">{content}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="msg-row right"><div class="msg assistant">{content}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Anzeige updaten
chat_box = st.container()
with chat_box:
    render_chat()

if user_msg and user_msg.strip():
    # Nutzer-Nachricht speichern und SOFORT anzeigen
    st.session_state["chat"].append({"role": "user", "content": user_msg.strip()})
    with chat_box:
        render_chat()

    # Systemprompt laden + API aufrufen
    system_prompt = render_sysmsg("Chatbot")
    payload = {
        "prompt": user_msg.strip(),
        "system_prompt": system_prompt,
        "temperature": st.session_state.get("temperature", 0.8),
        "top_p": st.session_state.get("top_p", 0.9),
        "max_tokens": 300,
    }

    try:
        with st.spinner("Antwort wird generiert …"):
            resp = session.post(f"{API_BASE_URL}/generate", json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json().get("response", "")
            if isinstance(data, list):
                data = "\n".join(data)
            reply = str(data)
        else:
            reply = f"API Error: {resp.status_code} - {resp.text}"
    except Exception as e:
        reply = f"Fehler beim Senden: {e}"

    # Assistant-Antwort speichern und NOCHMAL anzeigen
    st.session_state["chat"].append({"role": "assistant", "content": reply})
    with chat_box:
        render_chat()


# Zurück-Button
st.markdown("---")
if st.button("← Zurück", use_container_width=True):
    try:
        st.switch_page("streamlit.py")
    except Exception:
        st.query_params.clear()
        st.rerun()