"""Web UI for Scholarship Agent - Chat Interface with Hybrid LLM."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime

from src.chat_agent import ChatAgent
from src.email_sender import EmailSender, EmailConfig
from src.hybrid_llm import HybridLLM, LLMConfig

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
        
        # Model Status
        st.markdown("### 🤖 Model Status")
        llm_status = st.session_state.agent.llm.get_status()
        
        if llm_status['local_available']:
            st.success(f"Local: {llm_status['local_model']}")
        else:
            st.warning("Local Qwen: Not running")
        
        if llm_status['groq_configured']:
            st.success(f"Groq: {llm_status['groq_model']}")
        else:
            st.warning("Groq: Not configured")
        
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
        
        if st.button("🤖 Configure LLM", use_container_width=True):
            st.session_state.show_llm_setup = True
        
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

def llm_setup_modal():
    if st.session_state.get("show_llm_setup", False):
        with st.expander("🤖 Configure LLM Models", expanded=True):
            st.info("Configure local Qwen and/or Groq API for hybrid processing.")
            
            tab1, tab2 = st.tabs(["Groq API", "Local Qwen"])
            
            with tab1:
                st.subheader("Groq API (Free Tier)")
                st.markdown("""
                1. Go to [console.groq.com](https://console.groq.com)
                2. Create a free account
                3. Get your API key
                4. Paste it below
                """)
                
                groq_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    value=st.session_state.agent.llm.config.groq_api_key
                )
                
                groq_model = st.selectbox(
                    "Model",
                    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                    index=0
                )
                
                if st.button("Save Groq Config", type="primary", key="save_groq"):
                    st.session_state.agent.llm.config.groq_api_key = groq_key
                    st.session_state.agent.llm.config.groq_model = groq_model
                    st.session_state.agent.llm.config.save()
                    st.success("Groq configured!")
                    st.session_state.show_llm_setup = False
                    st.rerun()
            
            with tab2:
                st.subheader("Local Qwen 2.5 7B")
                st.markdown("""
                1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
                2. Pull model: `ollama pull qwen2.5:7b`
                3. Ollama runs on `localhost:11434` by default
                """)
                
                local_url = st.text_input(
                    "Ollama URL",
                    value=st.session_state.agent.llm.config.local_base_url
                )
                
                local_model = st.text_input(
                    "Model Name",
                    value=st.session_state.agent.llm.config.local_model
                )
                
                if st.button("Save Local Config", type="primary", key="save_local"):
                    st.session_state.agent.llm.config.local_base_url = local_url
                    st.session_state.agent.llm.config.local_model = local_model
                    st.session_state.agent.llm.config.save()
                    st.success("Local config saved!")
                    st.session_state.show_llm_setup = False
                    st.rerun()
            
            if st.button("Close", key="close_llm"):
                st.session_state.show_llm_setup = False
                st.rerun()

def main():
    st.title("🎓 Scholarship Application Agent")
    
    init_session_state()
    sidebar()
    
    cv_upload_section()
    smtp_setup_modal()
    llm_setup_modal()
    
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
