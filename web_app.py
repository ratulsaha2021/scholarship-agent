"""Minimalistic web UI for Scholarship Agent."""

import streamlit as st
from pathlib import Path

from src.chat_agent import ChatAgent

st.set_page_config(page_title="Scholar Agent", page_icon="🎓", layout="centered")

CUSTOM_CSS = """
<style>
    .stChatMessage { max-width: 800px; margin: auto; }
    .block-container { max-width: 900px; padding-top: 2rem; }
    header[data-testid="stHeader"] { display: none; }
    footer { display: none !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    return ChatAgent()


def init():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()


def sidebar():
    with st.sidebar:
        st.title("🎓 Scholar Agent")
        st.divider()

        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = load_agent()
            st.rerun()

        st.divider()

        llm = st.session_state.agent.llm.get_status()
        st.caption("Models")
        st.text("Local Llama " + ("✓" if llm["local_available"] else "✗"))
        st.text("Groq API " + ("✓" if llm["groq_configured"] else "✗"))
        st.text("Email " + ("✓" if st.session_state.agent.email_sender.is_configured() else "✗"))

        st.divider()

        uploaded = st.file_uploader(
            "Upload CV or Image",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

        if uploaded:
            ext = uploaded.name.split(".")[-1].lower()
            with st.spinner("Processing..."):
                resp = st.session_state.agent.handle_file_upload(
                    uploaded.name, uploaded.read(), ext
                )
            st.session_state.messages.append({"role": "assistant", "content": resp})
            st.rerun()


def main():
    init()
    sidebar()

    st.title("🎓 Scholar Agent")
    st.caption("Paste a post. I handle the rest.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Paste a post or ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("..."):
                resp = st.session_state.agent.chat(prompt)
            st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
