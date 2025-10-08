import streamlit as st

st.set_page_config(page_title='Chatbot', layout='wide')
st.title('Chatbot')

st.session_state.setdefault('korrigieren', False)
st.session_state.setdefault('bericht_typ', 'Chatbot')

# Steuerelemente
left, right = st.columns([2,8])
with left:
    st.checkout('Korrigieren', key='korrigieren')
    st.selectbox('Berichtstyp', ['', 'Chatbot'], key='bericht_typ')

# Textfeld
msg = st.text_area('Nachricht:', key='chat_input')

# Zurück zum Mainlayout
if st.button("← Zurück", use_container_width=True):
    st.session_state['bericht_typ'] = ""
    try:
        st.switch_page('streamlit.py')
    except Exception:
        st.experimental_set_query_params()
        st.rerun()