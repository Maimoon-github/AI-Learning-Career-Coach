import streamlit as st
import json
import asyncio
from datetime import datetime
from src.crewai_agents.profile_analysis_crew import ProfileAnalysisCrew
from src.crewai_agents.skill_gap_assessment_crew import SkillGapAssessmentCrew
from src.crewai_agents.learning_path_generation_crew import LearningPathGenerationCrew
from src.utils.llm_client import OllamaClient

# Set page config for a premium feel
st.set_page_config(
    page_title="Personalized AI Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism and premium aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e9ecef;
    }
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# App Navigation
st.sidebar.title("🤖 AI Career Coach")
menu = st.sidebar.radio("Navigate", ["Profile Analysis", "Skill Assessment", "Learning Path", "Coach Chat"])

# Session State for storing analysis results
if "profile_data" not in st.session_state:
    st.session_state.profile_data = None
if "skill_gaps" not in st.session_state:
    st.session_state.skill_gaps = None

def run_async(coro):
    return asyncio.run(coro)

# Main App Logic
if menu == "Profile Analysis":
    st.title("🌟 Professional Profile Analysis")
    st.markdown("Analyze your digital footprint across GitHub, Kaggle, and professional documents.")
    
    col1, col2 = st.columns(2)
    with col1:
        github_url = st.text_input("GitHub Profile URL", placeholder="https://github.com/username")
        kaggle_user = st.text_input("Kaggle Username", placeholder="username")
    with col2:
        resumes = st.file_uploader("Upload Resume/Portfolio (PDF)", accept_multiple_files=True)

    if st.button("Start Analysis"):
        with st.spinner("Our AI agents are analyzing your profiles..."):
            try:
                # In a real environment, we'd process files and urls
                crew = ProfileAnalysisCrew(
                    user_id="streamlit_user",
                    github_url=github_url if github_url else None,
                    kaggle_username=kaggle_user if kaggle_user else None
                )
                # We simulate/call the kickoff
                # For demo purposes, if postgres/ollama is missing, we show a friendly error
                if not OllamaClient().health_check():
                   st.error("Ollama service is not reachable. Please start Ollama to enable AI analysis.")
                else:
                    results = crew.kickoff()
                    st.session_state.profile_data = results
                    st.success("Analysis Complete!")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

    if st.session_state.profile_data:
        st.subheader("Analysis Insights")
        st.json(st.session_state.profile_data)

elif menu == "Skill Assessment":
    st.title("🎯 Skill Gap Assessment")
    target_role = st.selectbox("Select Target Role", ["ML Engineer", "Data Scientist", "Fullstack Developer", "AI Product Manager"])
    
    if st.button("Assess Gaps"):
        if not st.session_state.profile_data:
            st.warning("Please run Profile Analysis first!")
        else:
            with st.spinner("Comparing your profile with market requirements..."):
                crew = SkillGapAssessmentCrew(st.session_state.profile_data, target_role)
                results = crew.kickoff()
                st.session_state.skill_gaps = results
                st.success("Gaps Identified!")

    if st.session_state.skill_gaps:
        for gap in st.session_state.skill_gaps:
            with st.container():
                st.markdown(f"""
                <div class='card'>
                    <h3>{gap['skill_name']}</h3>
                    <p><b>Gap Severity:</b> {'🔴 High' if gap['gap_severity'] > 3 else '🟡 Medium'}</p>
                    <p>{gap['learning_objective']}</p>
                    <p><i>Weeks to close: {gap['weeks_to_close']}</i></p>
                </div>
                """, unsafe_allow_html=True)

elif menu == "Learning Path":
    st.title("🗺️ Your Personalized Learning Roadmap")
    if not st.session_state.skill_gaps:
        st.warning("Please complete Skill Assessment first!")
    else:
        if st.button("Generate Roadmap"):
            with st.spinner("Stitching together your path to success..."):
                crew = LearningPathGenerationCrew(st.session_state.skill_gaps)
                path = crew.kickoff()
                st.session_state.learning_path = path

        if "learning_path" in st.session_state:
            st.subheader(f"Roadmap to {st.session_state.learning_path.get('target_role', 'Success')}")
            for week in st.session_state.learning_path.get("weeks", []):
                with st.expander(f"Week {week['week_number']}: {week['primary_skill']}"):
                    st.write(f"**Topics:** {', '.join(week['topics'])}")
                    st.write(f"**Milestone:** {week['milestone']}")
                    st.write(f"**Estimated Hours:** {week['estimated_hours']}")

elif menu == "Coach Chat":
    st.title("💬 Talk to your AI Coach")
    st.info("The Voice interface is available via the main terminal app. Use this text interface for quick questions.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your career path..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = run_async(OllamaClient().generate(prompt, task_type="reasoning"))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"**System Status:** {'🟢 Healthy' if OllamaClient().health_check() else '🔴 Check Ollama'}")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
