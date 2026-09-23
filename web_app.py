"""Minimalistic web UI for Scholarship Agent."""

import streamlit as st

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


def md(text):
    """Keep single line breaks (Markdown would otherwise join the lines)."""
    return text.replace("\n", "  \n")


def init():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        # One agent per browser session: it holds conversation state (current post, draft)
        st.session_state.agent = ChatAgent()
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def sidebar():
    with st.sidebar:
        st.title("🎓 Scholar Agent")
        st.divider()

        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = ChatAgent()
            st.rerun()

        st.divider()

        llm = st.session_state.agent.llm.get_status()
        st.caption("Models")
        st.text("Local Llama " + ("✓" if llm["local_available"] else "✗"))
        st.text("Groq API " + ("✓" if llm["groq_configured"] else "✗"))
        st.text("Email " + ("✓" if st.session_state.agent.email_sender.is_configured() else "✗"))

        st.divider()

        docs = st.session_state.agent.saved_documents
        if docs:
            st.caption("Documents")
            for i, d in enumerate(reversed(docs[-4:])):
                path = d["paths"].get("docx") or d["paths"]["md"]
                mime = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        if path.suffix == ".docx" else "text/markdown")
                st.download_button(d["label"], path.read_bytes(), file_name=path.name, mime=mime,
                                   use_container_width=True, key=f"dl_{len(docs)}_{i}")
            st.divider()

        uploaded = st.file_uploader(
            "Upload CV or Image",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if uploaded:
            ext = uploaded.name.split(".")[-1].lower()
            with st.spinner("Processing..."):
                resp = st.session_state.agent.handle_file_upload(
                    uploaded.name, uploaded.read(), ext
                )
            st.session_state.messages.append({"role": "assistant", "content": resp})
            # Reset the uploader, otherwise the same file is processed again on every rerun
            st.session_state.uploader_key += 1
            st.rerun()


def main():
    init()
    sidebar()

    st.title("🎓 Scholar Agent")
    st.caption("Paste a post. I write the email, SOP and proposal.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(md(msg["content"]))

    if prompt := st.chat_input("Paste a post or ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(md(prompt))

        docs_before = len(st.session_state.agent.saved_documents)
        with st.chat_message("assistant"):
            with st.spinner("..."):
                resp = st.session_state.agent.chat(prompt)
            st.markdown(md(resp))

        st.session_state.messages.append({"role": "assistant", "content": resp})
        if len(st.session_state.agent.saved_documents) != docs_before:
            st.rerun()  # show the new document's download button in the sidebar


if __name__ == "__main__":
    main()
