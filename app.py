import streamlit as st
from chat import get_response

st.set_page_config(page_title="DIZI AI Chatbot", page_icon="🤖")
st.title("DIZI AI Chatbot (Streamlit)")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("You:", "", key="input")
if st.button("Send") and user_input.strip():
    response = get_response(user_input.strip(), username="streamlit_user")
    st.session_state['chat_history'].append(("You", user_input.strip()))
    st.session_state['chat_history'].append(("DIZI", response))

st.markdown("---")
for sender, msg in st.session_state['chat_history']:
    st.markdown(f"**{sender}:** {msg}")
