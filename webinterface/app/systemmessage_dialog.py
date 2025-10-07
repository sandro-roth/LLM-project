# systemmessage_dialog.py
import streamlit as st

def render_systemmessage_dialog():
    st.subheader("Systemmessage bearbeiten", divider="gray")

    # aktueller Override (falls vorhanden)
    current = st.session_state.get("system_message_override", "")
    st.text_area(
        "System Prompt",
        value=current,
        height=300,
        key="systemmessage_editor",
        help="Eigene Systemmessage, überschreibt die Auswahl aus system_messages.yml."
    )

    c1, c2 = st.columns(2)
    if c1.button("Übernehmen", use_container_width=True, key="apply_sysmsg"):
        st.session_state["system_message_override"] = st.session_state.get("systemmessage_editor", "")
        st.success("Übernommen – wird beim nächsten Generieren verwendet.")
    if c2.button("Zurücksetzen", use_container_width=True, key="reset_sysmsg"):
        st.session_state.pop("system_message_override", None)
        st.session_state.pop("systemmessage_editor", None)
        st.info("Zurückgesetzt – es wird wieder das YAML-Template genutzt.")
