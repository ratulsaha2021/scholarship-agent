"""Web UI for Scholarship Agent using Streamlit."""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime

from src.config import AgentConfig
from src.resource_loader import UserResources, create_sample_resources
from src.humanizer import Humanizer
from src.discovery import OpportunityDiscovery, Opportunity, create_sample_targets
from src.writer import EmailWriter, GeneratedEmail

RESOURCES_DIR = Path(__file__).parent / "resources"

st.set_page_config(
    page_title="Scholarship Agent",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_config():
    return AgentConfig.load()

@st.cache_resource
def load_humanizer():
    return Humanizer(level="high")

def load_resources():
    return UserResources.load(RESOURCES_DIR)

def init_session_state():
    if "generated_emails" not in st.session_state:
        st.session_state.generated_emails = []
    if "opportunities" not in st.session_state:
        st.session_state.opportunities = []
    if "resources_loaded" not in st.session_state:
        st.session_state.resources_loaded = False

def sidebar():
    with st.sidebar:
        st.title("🎓 Scholarship Agent")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Setup", "Discover", "Write Email", "Apply to Scholarship", "Saved Emails"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Generated Emails", len(st.session_state.generated_emails))
        
        return page

def setup_page():
    st.header("⚙️ Setup Your Profile")
    
    resources = load_resources()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        name = st.text_input("Full Name", value=resources.name)
        email = st.text_input("Email", value=resources.email)
        phone = st.text_input("Phone", value=resources.phone)
        
        st.subheader("Research Interests")
        interests_text = st.text_area(
            "Research interests (one per line)",
            value="\n".join(resources.research_interests),
            height=100,
            key="interests_text"
        )
        
        st.subheader("Skills")
        skills_text = st.text_area(
            "Skills (one per line)",
            value="\n".join(resources.skills),
            height=100,
            key="skills_text"
        )
    
    with col2:
        st.subheader("Upload CV")
        cv_file = st.file_uploader("Upload PDF or Word", type=["pdf", "docx", "txt"])
        
        if cv_file:
            cv_path = RESOURCES_DIR / f"cv{Path(cv_file.name).suffix}"
            cv_path.write_bytes(cv_file.read())
            st.success(f"CV saved: {cv_file.name}")
        
        st.subheader("Additional Notes")
        notes = st.text_area(
            "Any extra info you want included",
            value=resources.additional_notes,
            height=150,
            key="additional_notes"
        )
        
        st.subheader("Education")
        if resources.education:
            for i, edu in enumerate(resources.education):
                with st.expander(f"Education {i+1}"):
                    st.text(edu.get("degree", ""))
        
        if st.button("Add Education"):
            st.session_state.show_edu_form = True
        
        if st.session_state.get("show_edu_form", False):
            with st.form("edu_form"):
                degree = st.text_input("Degree (e.g., BSc, MSc)")
                field = st.text_input("Field of Study")
                institution = st.text_input("Institution")
                year = st.text_input("Year")
                submitted = st.form_submit_button("Save")
                
                if submitted:
                    edu = {"degree": degree, "field": field, "institution": institution, "year": year}
                    resources.education.append(edu)
                    st.session_state.show_edu_form = False
                    st.rerun()
    
    if st.button("Save Profile", type="primary"):
        user_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "research_interests": [i.strip() for i in interests_text.split("\n") if i.strip()],
            "skills": [s.strip() for s in skills_text.split("\n") if s.strip()],
            "education": resources.education,
            "experience": resources.experience,
            "publications": resources.publications,
            "awards": resources.awards,
            "target_universities": resources.target_universities
        }
        
        RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESOURCES_DIR / "user_data.json", "w") as f:
            json.dump(user_data, f, indent=2)
        
        if notes:
            (RESOURCES_DIR / "additional_notes.txt").write_text(notes)
        
        st.success("Profile saved!")
        st.session_state.resources_loaded = False

def discover_page():
    st.header("🔍 Discover Opportunities")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Search")
        query = st.text_input("Search for positions (e.g., 'machine learning PhD')")
        
        col_search, col_manual = st.columns(2)
        with col_search:
            search_online = st.checkbox("Search online", value=True)
        with col_manual:
            load_manual = st.checkbox("Load manual targets", value=True)
        
        if st.button("Search", type="primary"):
            discovery = OpportunityDiscovery()
            opportunities = []
            
            with st.spinner("Searching..."):
                if load_manual:
                    manual = discovery.load_manual_targets(RESOURCES_DIR / "targets.json")
                    opportunities.extend(manual)
                
                if search_online and query:
                    online = discovery.search_academic_positions(query)
                    opportunities.extend(online)
            
            st.session_state.opportunities = opportunities
            st.success(f"Found {len(opportunities)} opportunities")
    
    with col2:
        st.subheader("Add Manual Target")
        with st.form("manual_target"):
            title = st.text_input("Title")
            inst = st.text_input("Institution")
            opp_type = st.selectbox("Type", ["professor", "scholarship", "phd_position"])
            url = st.text_input("URL")
            email = st.text_input("Email (for professors)")
            deadline = st.text_input("Deadline")
            
            if st.form_submit_button("Add"):
                opp = Opportunity(
                    type=opp_type,
                    title=title,
                    institution=inst,
                    url=url,
                    professor_email=email,
                    deadline=deadline
                )
                st.session_state.opportunities.append(opp)
                st.rerun()
    
    if st.session_state.opportunities:
        st.subheader("Discovered Opportunities")
        
        for i, opp in enumerate(st.session_state.opportunities):
            with st.expander(f"{opp.type.upper()}: {opp.title} - {opp.institution}"):
                st.write(f"**Type:** {opp.type}")
                st.write(f"**Institution:** {opp.institution}")
                st.write(f"**URL:** {opp.url}")
                if opp.deadline:
                    st.write(f"**Deadline:** {opp.deadline}")
                if opp.professor_email:
                    st.write(f"**Email:** {opp.professor_email}")
                if opp.description:
                    st.write(f"**Description:** {opp.description}")
                
                if st.button("Select for Application", key=f"select_{i}"):
                    st.session_state.selected_opportunity = opp
                    st.rerun()

def write_email_page():
    st.header("✍️ Write Professor Email")
    
    resources = load_resources()
    humanizer = load_humanizer()
    writer = EmailWriter(resources, humanizer)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prof_name = st.text_input("Professor Name")
        prof_email = st.text_input("Professor Email")
        research_topic = st.text_input("Their Research Topic")
        
        st.subheader("Your Message")
        custom_message = st.text_area(
            "Anything specific you want to mention",
            placeholder="e.g., I read your paper on X and found Y particularly interesting...",
            key="custom_message"
        )
    
    with col2:
        st.subheader("Tips")
        st.info("""
        - Be specific about their work
        - Mention 1-2 specific papers
        - Connect their work to your interests
        - Keep it concise (200-300 words)
        - End with a clear ask
        """)
        
        st.subheader("Humanization Settings")
        level = st.select_slider("Humanization Level", options=["low", "medium", "high"], value="high")
        humanizer.level = level
    
    if st.button("Generate Email", type="primary"):
        if not prof_name or not prof_email:
            st.error("Please enter professor name and email")
            return
        
        with st.spinner("Generating humanized email..."):
            generated = writer.write_professor_email(
                prof_name, prof_email, research_topic, custom_message or None
            )
        
        st.session_state.generated_emails.append(generated)
        
        display_generated_email(generated)

def apply_page():
    st.header("📝 Apply to Scholarship")
    
    resources = load_resources()
    humanizer = load_humanizer()
    writer = EmailWriter(resources, humanizer)
    
    if not st.session_state.opportunities:
        st.warning("No opportunities loaded. Go to Discover page first.")
        return
    
    scholarships = [o for o in st.session_state.opportunities if o.type in ["scholarship", "phd_position"]]
    
    if not scholarships:
        st.warning("No scholarships found in discovered opportunities.")
        return
    
    selected = st.selectbox(
        "Select Scholarship",
        options=scholarships,
        format_func=lambda x: f"{x.title} - {x.institution}"
    )
    
    if selected:
        st.subheader(selected.title)
        st.write(f"**Institution:** {selected.institution}")
        st.write(f"**Deadline:** {selected.deadline or 'N/A'}")
        if selected.description:
            st.write(f"**Description:** {selected.description}")
        
        additional = st.text_area("Additional information to include", key="additional_info")
        
        if st.button("Generate Application", type="primary"):
            with st.spinner("Generating application..."):
                generated = writer.write_scholarship_application(selected, additional or None)
            
            st.session_state.generated_emails.append(generated)
            display_generated_email(generated)

def display_generated_email(email: GeneratedEmail):
    st.markdown("---")
    st.subheader("Generated Email")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.text_input("Subject", value=email.subject, disabled=True, key=f"subject_{email.to_email}")
        st.text_input("To", value=email.to_email, disabled=True, key=f"to_{email.to_email}")
        st.text_area("Email Body", value=email.body, height=400, disabled=True, key=f"body_{email.to_email}")
    
    with col2:
        st.subheader("Review")
        review = EmailWriter(load_resources(), load_humanizer()).review_email(email)
        
        st.metric("Word Count", review["word_count"])
        st.metric("Humanization Score", f"{review['humanization_score']:.2f}")
        st.metric("AI Patterns Found", review["ai_patterns_found"])
        
        if review["changes_made"]:
            st.write("**Changes Made:**")
            for change in review["changes_made"]:
                st.write(f"- {change}")
        
        st.download_button(
            "Download Email",
            data=email.body,
            file_name=f"email_{email.to_email.split('@')[0]}.txt",
            mime="text/plain"
        )

def saved_page():
    st.header("💾 Saved Emails")
    
    if not st.session_state.generated_emails:
        st.info("No emails generated yet.")
        return
    
    st.write(f"**Total:** {len(st.session_state.generated_emails)} emails")
    
    for i, email in enumerate(st.session_state.generated_emails):
        with st.expander(f"Email to {email.to_email}"):
            st.write(f"**Subject:** {email.subject}")
            st.write(f"**Humanization:** {email.humanization_result.confidence_score:.2f}")
            st.text_area(
                "Body",
                value=email.body,
                height=200,
                key=f"email_{i}",
                disabled=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download",
                    data=email.body,
                    file_name=f"email_{i+1}.txt",
                    key=f"download_{i}"
                )
            with col2:
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.generated_emails.pop(i)
                    st.rerun()
    
    if st.button("Export All", type="primary"):
        export = []
        for email in st.session_state.generated_emails:
            export.append({
                "to": email.to_email,
                "subject": email.subject,
                "body": email.body,
                "humanization_score": email.humanization_result.confidence_score
            })
        
        st.download_button(
            "Download All Emails (JSON)",
            data=json.dumps(export, indent=2),
            file_name="all_emails.json",
            mime="application/json"
        )

def main():
    init_session_state()
    page = sidebar()
    
    if page == "Setup":
        setup_page()
    elif page == "Discover":
        discover_page()
    elif page == "Write Email":
        write_email_page()
    elif page == "Apply to Scholarship":
        apply_page()
    elif page == "Saved Emails":
        saved_page()

if __name__ == "__main__":
    main()
