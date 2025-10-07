import streamlit as st
from jinja2 import Template

def render_systemmessage_dialog():
    st.markdown("""
        <style>
        /* gesamter Dialogbereich breiter machen */
        section[data-testid="stPopoverBody"], section[data-testid="stDialog"] {
            width: 900px !important;      /* Gesamtbreite */
            max-width: 95vw !important;   /* auf kleinen Screens begrenzen */
        }

        /* Textarea speziell in diesem Dialog */
        textarea[aria-label="System Prompt"] {
            width: 100% !important;
            min-width: 850px !important;
            height: 500px !important;
            font-size: 0.9rem;
            line-height: 1.4;
            resize: vertical;
        }
        </style>
        """, unsafe_allow_html=True)

    st.subheader("Systemmessage bearbeiten", divider="gray")

    # aktueller Override (falls vorhanden)
    current = st.session_state.get("system_message_override", "")
    st.text_area(
        "System Prompt",
        value=current,
        height=500,
        key="systemmessage_editor",
        help="Eigene Systemmessage, überschreibt die Auswahl aus system_messages.yml."
    )

    c1, c2, c3 = st.columns(2)
    if c1.button('Show', use_container_width=True, key='show_sysmsg'):
        effective = None
        effective = st.session_state.get('rendered_system_message')

        if not effective:
            effective = st.session_state.get('system_message_override')

        if not effective:
            tpl = st.sesseion_state.get('active_system_template')
            ctx = st.sessions_state.get('active_system_context', {}) or {}
            if tpl:
                try:
                    effective = Template(tpl).render(ctx)
                except Exception:
                    effective = tpl

        # Fallback: leeren string
        st.session_state['systemmessage_editor'] = effective or ""
        st.toast('Aktueller Systemprompt geladen')
        st.rerun()

    if c2.button("Übernehmen", use_container_width=True, key="apply_sysmsg"):
        st.session_state["system_message_override"] = st.session_state.get("systemmessage_editor", "")
        st.success("Übernommen – wird beim nächsten Generieren verwendet.")

    if c3.button("Zurücksetzen", use_container_width=True, key="reset_sysmsg"):
        st.session_state.pop("system_message_override", None)
        st.session_state.pop("systemmessage_editor", None)
        st.info("Zurückgesetzt – es wird wieder das YAML-Template genutzt.")
