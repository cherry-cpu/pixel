import os
import streamlit as st

from pixel_app.ui.pages import (
    page_library,
    page_people,
    page_search,
    page_share,
    page_settings,
)
from pixel_app.core.app_state import get_app


st.set_page_config(
    page_title="Pixel — AI Photo Memory Manager",
    page_icon="📷",
    layout="wide",
)


def main() -> None:
    app = get_app()

    with st.sidebar:
        st.title("Pixel")
        st.caption("Encrypted local photo library with face AI + NL search.")

        passphrase = st.text_input(
            "Library passphrase",
            type="password",
            help="Used to encrypt/decrypt your stored photos. Keep it safe.",
        )
        if passphrase:
            ok, msg = app.auth.ensure_unlocked(passphrase)
            if ok:
                st.success("Library unlocked")
            else:
                st.error(msg)
        else:
            st.info("Enter your passphrase to unlock.")

        st.divider()
        page = st.radio(
            "Navigate",
            ["Library", "People", "Search", "Share", "Settings"],
            index=0,
        )

        st.divider()
        st.caption("LLM providers")
        st.write(
            {
                "groq": "configured" if os.getenv("GROQ_API_KEY") else "not set",
                "huggingface": "configured" if os.getenv("HF_TOKEN") else "not set",
            }
        )

    if page == "Library":
        page_library(app)
    elif page == "People":
        page_people(app)
    elif page == "Search":
        page_search(app)
    elif page == "Share":
        page_share(app)
    else:
        page_settings(app)


if __name__ == "__main__":
    main()

