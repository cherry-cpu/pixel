import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pixel_app.ui.pages import (
    page_library,
    page_library,
    page_people,
    page_search,
    page_share,
    page_settings,
    page_dashboard,
    page_chatbot,
)
from pixel_app.core.app_state import get_app


st.set_page_config(
    page_title="Pixel — AI Photo Memory Manager",
    page_icon="📷",
    layout="wide",
)


def render_auth(app) -> None:
    st.title("Welcome to Pixel")
    st.caption("Please log in or sign up to access your memory manager.")
    
    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        l_username = st.text_input("Username", key="login_username")
        l_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In"):
            if app.auth.verify_user(l_username, l_password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = l_username
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    with tab2:
        st.subheader("Create an Account")
        s_username = st.text_input("New Username", key="signup_username")
        s_password = st.text_input("New Password", type="password", key="signup_password")
        s_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        
        if st.button("Sign Up"):
            if not s_username or not s_password:
                st.error("Please provide both username and password")
            elif s_password != s_password_confirm:
                st.error("Passwords do not match")
            else:
                success, msg = app.auth.register_user(s_username, s_password)
                if success:
                    st.success("Account created successfully! Please log in.")
                else:
                    st.error(msg)


def main() -> None:
    app = get_app()
    app.auth.ensure_initialized() # Make sure meta tables are created if empty

    if not st.session_state.get("logged_in"):
        render_auth(app)
        return

    with st.sidebar:
        st.title("Pixel")
        st.caption(f"Logged in as {st.session_state.get('username')}")
        st.caption("Encrypted local photo library with face AI + NL search.")

        st.divider()
        page = st.radio(
            "Navigate",
            ["Dashboard", "Library", "People", "Search", "Chatbot", "Share", "Settings"],
            index=0,
        )
        if st.button("Logout"):
            st.session_state["logged_in"] = False
            if "username" in st.session_state:
                del st.session_state["username"]
            st.rerun()

    if page == "Dashboard":
        page_dashboard(app)
    elif page == "Library":
        page_library(app)
    elif page == "People":
        page_people(app)
    elif page == "Search":
        page_search(app)
    elif page == "Chatbot":
        page_chatbot(app)
    elif page == "Share":
        page_share(app)
    else:
        page_settings(app)


if __name__ == "__main__":
    main()

