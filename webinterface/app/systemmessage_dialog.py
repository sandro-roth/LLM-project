import streamlit as st
from jinja2 import Template

def render_systemmessage_dialog(active_key: str, get_effective_system_message):
    """
        Rendert das Systemmessage-Popover für den gegebenen Berichtstyp.
        - active_key: z. B. 'Korrigieren' oder ein Berichtstyp aus YAML
        - get_effective_system_message: callable(key) -> str, liefert gerenderten Prompt aus YAML
    """
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

    # State-Container für alle Overrides
    if 'sysmsg_overrides' not in st.session_state:
        st.session_state['sysmsg_overrides'] = {}

    overrides = st.session_state['sysmsg_overrides']
    editor_key = f"systemmessage_editor__{active_key}"

    # aktueller Override (falls vorhanden)

    current = overrides.get(active_key, "")
    st.text_area(
        f'Systemprompt für: {active_key}',
        key=editor_key,
        value=current,
        height=500,
        help="Eigene Systemmessage, überschreibt die Auswahl aus system_messages.yml."
    )

    c1, c2, c3 = st.columns(3)
    if c1.button('Show', use_container_width=True, key=f'show_sysmsg__{active_key}'):
        override = overrides.get(active_key, "").strip()
        effective = overrides if override else (get_effective_system_message(active_key) or "")
        st.session_state[editor_key] = effective
        st.toast('Aktueller Systemprompt geladen')
        st.rerun()

    if c2.button("Speichern", use_container_width=True, key=f"apply_sysmsg__{active_key}"):
        overrides[active_key] = (st.session_state.get(editor_key) or "").strip()
        st.success(f"Systemmessage für {active_key} gespeichert (Override aktiv).")

    if c3.button("Zurücksetzen", use_container_width=True, key=f"reset_sysmsg__{active_key}"):
        overrides.pop(active_key, None)
        st.session_state.pop(editor_key, None)
        st.info(f'Override für {active_key} entfernt - YAML-Template wird wieder verwendet.')
        st.rerun()
