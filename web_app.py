"""Web UI for Scholar Agent."""

import streamlit as st

from src.chat_agent import ChatAgent

st.set_page_config(page_title="Scholar Agent", page_icon=":material/school:", layout="centered")

USER_AVATAR = ":material/person:"
AGENT_AVATAR = ":material/school:"
UPLOAD_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg"]

SUGGESTIONS = {
    ":material/travel_explore: Find ML PhD positions": "find machine learning phd",
    ":material/person: Show my profile": "status",
    ":material/notifications: What's new": "digest",
    ":material/help: What can you do?": "help",
}

STEPS = [
    (":material/travel_explore:", "Find a post",
     "Paste a post or link, attach a screenshot, or search with `find ...`."),
    (":material/edit_note:", "Write",
     "Email, statement of purpose and research proposal, fact-checked against your CV."),
    (":material/send:", "Review and send",
     "Edit in plain words. Nothing is sent until you type `send`."),
]


def md(text):
    """Keep single line breaks (Markdown would otherwise join the lines)."""
    return text.replace("\n", "  \n")


def init():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        # One agent per browser session: it holds conversation state (current post, draft)
        st.session_state.agent = ChatAgent()
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("queued", None)


def spinner_text(prompt):
    p = prompt.lower().strip()
    if p.startswith(("find", "search", "look for", "digest")) or p.startswith("http"):
        return "Searching job sites..."
    if p.startswith(("write", "draft", "apply", "sop", "skip")) or len(p.split()) > 3:
        return "Writing, then checking style and facts. This can take a minute or two..."
    return "Thinking..."


def run(prompt, files=()):
    """Send text and/or files to the agent and record both sides of the exchange."""
    agent = st.session_state.agent
    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        st.session_state.messages.append({"role": "user", "content": f":material/attach_file: `{f.name}`"})
        with st.spinner(f"Reading {f.name}..."):
            reply = agent.handle_file_upload(f.name, f.getvalue(), ext)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(md(prompt))
        with st.chat_message("assistant", avatar=AGENT_AVATAR):
            with st.spinner(spinner_text(prompt)):
                reply = agent.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


def badge(ok, on, off, icon_on="check_circle", icon_off="radio_button_unchecked"):
    """Inline badge markup; several in one st.markdown sit on one line."""
    color, label, icon = ("green", on, icon_on) if ok else ("gray", off, icon_off)
    return f":{color}-badge[:material/{icon}: {label}]"


def sidebar():
    agent = st.session_state.agent
    res = agent.resources
    with st.sidebar:
        st.markdown("### :material/school: Scholar Agent")
        st.caption("PhD and scholarship applications, start to finish.")

        if st.button("New chat", icon=":material/add_comment:", width="stretch"):
            st.session_state.messages = []
            st.session_state.agent = ChatAgent()
            st.rerun()

        with st.container(border=True):
            st.markdown("**:material/badge: Your profile**")
            if res.name:
                name = res.name.title() if res.name.isupper() else res.name
                st.markdown(f"{name}  \n:gray[{res.email}]")
            else:
                st.caption("No CV yet. Upload one to get started.")
            st.markdown(badge(agent._cv_file() is not None, "CV on file", "No CV") + " "
                        + badge(agent._sop_notes_file().exists(), "SOP notes saved", "No SOP notes"))
            has_cv = agent._cv_file() is not None
            with st.expander("Replace CV" if has_cv else "Upload CV", icon=":material/upload_file:",
                             expanded=not has_cv):
                cv = st.file_uploader("CV file", type=["pdf", "docx", "txt"],
                                      key=f"cv_{st.session_state.uploader_key}", label_visibility="collapsed")
            if cv:
                st.session_state.uploader_key += 1  # reset, or the file is re-read on every rerun
                run("", [cv])

        post = agent.current_post
        if post:
            with st.container(border=True):
                st.markdown("**:material/work: Current application**")
                st.markdown(f"{post['title']}")
                details = " · ".join(p for p in [post.get("institution"), post.get("deadline")] if p)
                if details:
                    st.caption(details)
                st.markdown(" ".join([
                    badge(agent.pending_email is not None, "Email", "Email"),
                    badge("sop" in agent.post_documents, "SOP", "SOP"),
                    badge("proposal" in agent.post_documents, "Proposal", "Proposal"),
                ]))

        docs = agent.saved_documents
        if docs:
            with st.container(border=True):
                st.markdown("**:material/folder_open: Documents**")
                for i, d in enumerate(reversed(docs[-4:])):
                    path = d["paths"].get("docx") or d["paths"]["md"]
                    mime = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            if path.suffix == ".docx" else "text/markdown")
                    st.download_button(d["label"], path.read_bytes(), file_name=path.name, mime=mime,
                                       icon=":material/download:", type="tertiary",
                                       key=f"dl_{len(docs)}_{i}")

        llm = agent.llm.get_status()
        st.caption("Connections")
        st.markdown(" ".join([
            badge(llm["groq_configured"], "Groq", "Groq off", "cloud_done", "cloud_off"),
            badge(llm["local_available"], "Local model", "Local off", "memory", "memory"),
            badge(agent.email_sender.is_configured(), "Email", "Email off", "mail", "mail"),
        ]))
        if not agent.email_sender.is_configured():
            st.caption("Type `setup email gmail` to enable sending.")
        st.caption(":material/contrast: Light or dark theme: menu **⋮** (top right).")


def welcome():
    st.space("small")
    st.markdown("Paste a scholarship or PhD post and I'll handle the application: "
                "a tailored email, a statement of purpose and a research proposal, "
                "written from your CV and checked for AI-sounding phrasing and wrong facts.")
    with st.container(horizontal=True, gap="medium"):
        for icon, title, text in STEPS:
            with st.container(border=True):
                st.markdown(f"**{icon} {title}**")
                st.caption(text)
    st.space("small")
    choice = st.pills("Try", list(SUGGESTIONS), label_visibility="collapsed", key="suggestion")
    if choice:
        st.session_state.queued = SUGGESTIONS[choice]


def action_bar():
    """One-click commands for the loaded post."""
    agent = st.session_state.agent
    if not agent.current_post or agent.flow_state in ("waiting_sop_notes", "waiting_proposal_topic"):
        return
    with st.container(horizontal=True, gap="small"):
        if st.button("Write email", icon=":material/mail:", type="secondary"):
            st.session_state.queued = "write"
        if st.button("Write SOP", icon=":material/description:", type="secondary"):
            st.session_state.queued = "write sop"
        if st.button("Write proposal", icon=":material/science:", type="secondary"):
            st.session_state.queued = "write proposal"
        if agent.pending_email and st.button("Send email", icon=":material/send:", type="primary"):
            st.session_state.queued = "send"


def main():
    init()
    sidebar()

    st.title("Scholar Agent")
    if not st.session_state.messages:
        welcome()

    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(md(msg["content"]))

    action_bar()

    submitted = st.chat_input(
        "Paste a post or link, type find ..., or attach your CV",
        accept_file=True, file_type=UPLOAD_TYPES, submit_mode="disable",
    )
    if st.session_state.queued:
        prompt, st.session_state.queued = st.session_state.queued, None
        run(prompt)
    elif submitted:
        run(submitted.text.strip(), submitted.files)


if __name__ == "__main__":
    main()
