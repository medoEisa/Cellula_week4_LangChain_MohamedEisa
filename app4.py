import streamlit as st
import json
from auth import create_users_table, add_user, verify_user, load_conversation, save_conversation
from rag_handler import get_rag_response

st.set_page_config(page_title=" RAG Chat", page_icon="🎓", layout="wide")

# Initialize DB
create_users_table()

# Session states
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# LOGIN / SIGNUP 
def login_ui():
    st.title(" RAG Chat")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.subheader("Login to your account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.chat_history = json.loads(load_conversation(username))
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Create a new account")
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            try:
                add_user(new_user, new_pass)
                st.success("Account created! You can log in now.")
            except:
                st.error("Username already exists.")

# CHAT INTERFACE 
def chat_ui():
    st.title(f" AI Assistant — Welcome {st.session_state.username}")
    st.markdown("---")

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_query = st.chat_input("Ask something about AI, ML, etc...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_rag_response(user_query)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        save_conversation(st.session_state.username, json.dumps(st.session_state.chat_history))

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.chat_history = []
        st.rerun()

#  MAIN 
if not st.session_state.logged_in:
    login_ui()
else:
    chat_ui()
