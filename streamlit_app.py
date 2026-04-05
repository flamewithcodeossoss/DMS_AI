import streamlit as st
import requests
import json
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────
# MODAL BACKEND URL
# When deployed on Modal the MODAL_API_URL env var is injected
# automatically. The hardcoded fallback is used for local dev.
# ─────────────────────────────────────────────────────────────
import os
MODAL_API_URL = os.environ.get(
    "MODAL_API_URL",
    "https://waitforossoss14--medical-rag-dms-fastapi-app.modal.run",
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG DMS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .summary-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .insights-box {
        background-color: #f0f8ff;
        border-left: 4px solid #2ca02c;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffe8e8;
        border-left: 4px solid #d62728;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #2ca02c;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# APP TITLE & DESCRIPTION
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏥 Medical RAG Document Analysis System</div>', unsafe_allow_html=True)
st.markdown("### Analyze medical documents using AI-powered insights")

# ─────────────────────────────────────────────────────────────
# SIDEBAR - CONFIGURATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Backend (Modal — read-only) ──────────────────────────
    api_url = MODAL_API_URL
    st.markdown("**Backend (Modal)**")
    st.code(api_url, language=None)
    st.caption("Hosted on Modal serverless infrastructure.")

    st.divider()

    # Session ID
    session_id = st.text_input(
        "Session ID",
        value="default_session",
        help="Unique identifier for this analysis session"
    )

    st.divider()
    st.markdown("### About")
    st.info("""
    **Medical RAG DMS** is a document analysis system that:
    - 📄 Extracts medical insights from PDFs, images, and text
    - 🧠 Uses AI to understand clinical information
    - 💾 Maintains conversation history per session
    - 🔍 Provides structured JSON responses
    """)

    st.markdown(
        f"[📖 API Docs]({api_url}/docs)  •  [❤️ Health]({api_url}/health)",
        unsafe_allow_html=False,
    )

# ─────────────────────────────────────────────────────────────
# MAIN CONTENT - INPUT SECTION
# ─────────────────────────────────────────────────────────────
st.header("📥 Upload or Enter Medical Document")

# Tabs for different input types
tab1, tab2, tab3 = st.tabs(["📄 PDF Upload", "🖼️ Image Upload", "📝 Text Input"])

with tab1:
    st.subheader("Upload a PDF File")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload")
    
    if pdf_file and st.button("Analyze PDF", key="btn_analyze_pdf", use_container_width=True):
        st.session_state.file_to_analyze = ("pdf", pdf_file)
        st.session_state.input_type = "PDF"

with tab2:
    st.subheader("Upload an Image")
    image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "gif"], key="image_upload")
    
    if image_file:
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
    
    if image_file and st.button("Analyze Image", key="btn_analyze_image", use_container_width=True):
        st.session_state.file_to_analyze = ("image", image_file)
        st.session_state.input_type = "IMAGE"

with tab3:
    st.subheader("Enter Medical Text")
    text_input = st.text_area(
        "Paste medical document text here",
        height=200,
        placeholder="Enter medical document text, clinical notes, lab results, etc.",
        key="text_input"
    )
    
    if text_input and st.button("Analyze Text", key="btn_analyze_text", use_container_width=True):
        st.session_state.text_to_analyze = text_input
        st.session_state.input_type = "TEXT"

# ─────────────────────────────────────────────────────────────
# PROCESS UPLOAD
# ─────────────────────────────────────────────────────────────
if "file_to_analyze" in st.session_state or "text_to_analyze" in st.session_state:
    st.divider()
    st.header("⏳ Processing...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Prepare request
        status_text.text("Preparing request...")
        progress_bar.progress(20)
        
        analyze_url = f"{api_url}/api/v1/analyze"
        headers = {"accept": "application/json"}
        
        # Determine input type and prepare data
        if "file_to_analyze" in st.session_state:
            file_type, file_obj = st.session_state.file_to_analyze
            files = {
                "file": (file_obj.name, file_obj.getbuffer(), file_obj.type),
                "session_id": (None, session_id),
            }
            status_text.text(f"Uploading {file_type}...")
            progress_bar.progress(40)
            
        elif "text_to_analyze" in st.session_state:
            text_content = st.session_state.text_to_analyze
            data = {
                "text": text_content,
                "session_id": session_id,
            }
            files = None
            status_text.text("Sending text to analysis...")
            progress_bar.progress(40)
        
        # Send request
        status_text.text("Analyzing document with AI...")
        progress_bar.progress(60)
        
        if files:
            response = requests.post(analyze_url, files=files, headers=headers, timeout=300)
        else:
            response = requests.post(analyze_url, data=data, headers=headers, timeout=300)
        
        progress_bar.progress(80)
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            st.session_state.analysis_result = result
            
            # Clean up
            if "file_to_analyze" in st.session_state:
                del st.session_state.file_to_analyze
            if "text_to_analyze" in st.session_state:
                del st.session_state.text_to_analyze
            
            progress_bar.progress(100)
            status_text.success("✅ Analysis Complete!")
            
        else:
            error_msg = response.json().get("detail", response.text)
            st.markdown(f'<div class="error-box"><strong>❌ Error:</strong> {error_msg}</div>', unsafe_allow_html=True)
            status_text.error(f"API Error: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.markdown(
            f'<div class="error-box"><strong>❌ Connection Error:</strong> '
            f'Cannot reach the Modal backend at <code>{api_url}</code>. '
            f'Verify the deployment is live by running <code>modal app list</code> '
            f'or visiting the <a href="{api_url}/health" target="_blank">health endpoint</a>.</div>',
            unsafe_allow_html=True,
        )
    except requests.exceptions.Timeout:
        st.markdown('<div class="error-box"><strong>❌ Timeout:</strong> Analysis took too long. Please try again.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error-box"><strong>❌ Unexpected Error:</strong> {str(e)}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────
if "analysis_result" in st.session_state:
    result = st.session_state.analysis_result
    insights = result.get("insights", {})
    
    st.divider()
    st.header("📋 Analysis Results")
    
    # ── HUMAN SUMMARY (PROMINENT) ──
    if insights.get("human_summary"):
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown("### 📝 Clinical Summary")
        st.markdown(f"**{insights['human_summary']}**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── TABS FOR DETAILED INSIGHTS ──
    col1, col2 = st.columns(2)
    
    with col1:
        # PATIENT INFO
        if insights.get("patient_info"):
            with st.expander("👤 Patient Information", expanded=True):
                patient = insights["patient_info"]
                
                info_cols = st.columns(2)
                with info_cols[0]:
                    if patient.get("name"):
                        st.metric("Name", patient["name"])
                    if patient.get("age"):
                        st.metric("Age", patient["age"])
                
                with info_cols[1]:
                    if patient.get("gender"):
                        st.metric("Gender", patient["gender"])
                    if patient.get("id"):
                        st.metric("ID", patient["id"])
                
                if patient.get("date_of_report"):
                    st.write(f"**Date of Report:** {patient['date_of_report']}")
    
    with col2:
        # SYMPTOMS
        if insights.get("symptoms"):
            with st.expander("🩺 Symptoms", expanded=True):
                symptoms = insights["symptoms"]
                for symptom in symptoms:
                    symptom_text = f"**{symptom.get('name', 'Unknown')}**"
                    if symptom.get("severity"):
                        symptom_text += f" (Severity: {symptom['severity']})"
                    if symptom.get("duration"):
                        symptom_text += f"\n*Duration: {symptom['duration']}*"
                    if symptom.get("notes"):
                        symptom_text += f"\n📌 {symptom['notes']}"
                    st.markdown(symptom_text)
    
    # MEDICAL ASSESSMENT
    if insights.get("medical_assessment"):
        st.markdown("---")
        with st.expander("🔬 Medical Assessment", expanded=True):
            assessment = insights["medical_assessment"]
            
            if assessment.get("diagnosis"):
                st.markdown(f"**Diagnosis:** {assessment['diagnosis']}")
            
            if assessment.get("findings"):
                st.markdown("**Key Findings:**")
                for finding in assessment["findings"]:
                    st.markdown(f"- {finding}")
            
            if assessment.get("medications"):
                st.markdown("**Medications:**")
                for med in assessment["medications"]:
                    st.markdown(f"- {med}")
            
            if assessment.get("lab_results"):
                st.markdown("**Lab Results:**")
                for key, value in assessment["lab_results"].items():
                    st.markdown(f"- **{key}:** {value}")
            
            if assessment.get("imaging_results"):
                st.markdown(f"**Imaging Results:** {assessment['imaging_results']}")
    
    # SUGGESTED NEXT STEPS
    if insights.get("suggested_next_steps"):
        st.markdown("---")
        with st.expander("📋 Suggested Next Steps", expanded=True):
            for step in insights["suggested_next_steps"]:
                priority = step.get("priority", "").upper() if step.get("priority") else "N/A"
                priority_emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
                
                st.markdown(f"{priority_emoji} **{step.get('action', 'Unknown')}** *(Priority: {priority})*")
                if step.get("reason"):
                    st.markdown(f"> {step['reason']}")
    
    # RAW JSON (for developers)
    st.divider()
    with st.expander("🔧 Raw JSON Response"):
        st.json(result)
    
    # Metadata
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input Type", result.get("input_type", "Unknown"))
    with col2:
        st.metric("Chunks Indexed", result.get("chunks_indexed", 0))
    
    # Clear button
    if st.button("Clear Results", key="btn_clear", use_container_width=True):
        del st.session_state.analysis_result
        st.rerun()

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
    <p>🏥 Medical RAG DMS &nbsp;•&nbsp; UI: Streamlit &nbsp;•&nbsp; Backend: <a href="{MODAL_API_URL}" target="_blank">Modal</a></p>
    <p>API docs: <a href="{MODAL_API_URL}/docs" target="_blank">{MODAL_API_URL}/docs</a></p>
</div>
""", unsafe_allow_html=True)
