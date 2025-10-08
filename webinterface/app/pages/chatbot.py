import streamlit as st

st.set_page_config(page_title='Chatbot', layout='wide')
st.title('Chatbot')

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