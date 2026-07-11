"""Web UI for Scholarship Agent - Chat Interface."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime

from src.chat_agent import ChatAgent
from src.email_sender import EmailSender, EmailConfig

st.set_page_config(
    page_title="Scholarship Agent",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_agent():
    return ChatAgent()

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()

def sidebar():
    with st.sidebar:
        st.title("🎓 Scholarship Agent")
        st.markdown("---")
        
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = load_agent()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("📄 Upload CV", use_container_width=True):
            st.session_state.show_cv_upload = True
        
        if st.button("🔍 Search Posts", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "search posts"
            })
            response = st.session_state.agent.chat("search posts")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        
        if st.button("✉️ Write Email", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "write email"
            })
            response = st.session_state.agent.chat("write email")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        
        if st.button("❓ Help", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "help"
            })
            response = st.session_state.agent.chat("help")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Settings")
        
        if st.button("⚙️ Setup Email", use_container_width=True):
            st.session_state.show_smtp_setup = True
        
        st.markdown("---")
        st.markdown("### Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        agent = st.session_state.agent
        rag_stats = agent.rag_store.get_stats()
        st.metric("Saved Posts", rag_stats["total_posts"])

def display_chat():
    st.header("💬 Chat with Scholarship Agent")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Tell me what you need..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

def cv_upload_section():
    if st.session_state.get("show_cv_upload", False):
        with st.expander("📄 Upload CV", expanded=True):
            cv_file = st.file_uploader(
                "Upload your CV",
                type=["pdf", "docx", "txt"],
                key="cv_upload_main"
            )
            
            if cv_file:
                file_type = cv_file.name.split(".")[-1].lower()
                
                with st.spinner("Processing CV..."):
                    response = st.session_state.agent.handle_file_upload(
                        cv_file.name,
                        cv_file.read(),
                        file_type
                    )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.session_state.show_cv_upload = False
                st.rerun()
            
            if st.button("Cancel", key="cancel_cv"):
                st.session_state.show_cv_upload = False
                st.rerun()

def image_upload_section():
    uploaded_file = st.file_uploader(
        "📷 Upload Image of Post",
        type=["png", "jpg", "jpeg"],
        key="image_upload_chat"
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        with st.spinner("Extracting text from image..."):
            response = st.session_state.agent.handle_file_upload(
                uploaded_file.name,
                uploaded_file.read(),
                uploaded_file.name.split(".")[-1].lower()
            )
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        st.rerun()

def smtp_setup_modal():
    if st.session_state.get("show_smtp_setup", False):
        with st.expander("⚙️ Setup Email (SMTP)", expanded=True):
            st.info("Configure email sending. For Gmail, use an App Password.")
            
            provider = st.selectbox(
                "Email Provider",
                ["Gmail", "Outlook", "Yahoo", "Other"]
            )
            
            if provider == "Gmail":
                server = "smtp.gmail.com"
                port = 587
            elif provider == "Outlook":
                server = "smtp-mail.outlook.com"
                port = 587
            elif provider == "Yahoo":
                server = "smtp.mail.yahoo.com"
                port = 587
            else:
                server = st.text_input("SMTP Server")
                port = st.number_input("SMTP Port", value=587)
            
            email = st.text_input("Email Address")
            password = st.text_input("Password or App Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save & Test", type="primary"):
                    if email and password:
                        config = EmailConfig(
                            smtp_server=server,
                            smtp_port=port,
                            email_address=email,
                            password=password
                        )
                        config.save()
                        
                        sender = EmailSender(config)
                        result = sender.test_connection()
                        
                        if result['success']:
                            st.success("Email configured successfully!")
                            st.session_state.show_smtp_setup = False
                            st.rerun()
                        else:
                            st.error(f"Connection failed: {result['error']}")
                    else:
                        st.warning("Please fill in all fields")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.show_smtp_setup = False
                    st.rerun()

def main():
    st.title("🎓 Scholarship Application Agent")
    
    init_session_state()
    sidebar()
    
    cv_upload_section()
    smtp_setup_modal()
    
    col_chat, col_upload = st.columns([3, 1])
    
    with col_chat:
        display_chat()
    
    with col_upload:
        image_upload_section()
        
        st.markdown("---")
        st.markdown("### Or tell me what to do:")
        
        examples = [
            "Upload my CV",
            "Add this scholarship post",
            "Write email to professor",
            "Apply for PhD position",
            "Search my saved posts",
            "Help"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": example
                })
                response = st.session_state.agent.chat(example)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()

if __name__ == "__main__":
    main()
