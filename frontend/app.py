"""
SafeRx Frontend - Streamlit Application
AI Medical Prescription Verification System
"""

import streamlit as st
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"


# Page Configuration
st.set_page_config(
    page_title="SafeRx - AI Prescription Verifier",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-badge {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #4CAF50;
        color: white;
    }
    .risk-medium {
        background-color: #FF9800;
        color: white;
    }
    .risk-high {
        background-color: #F44336;
        color: white;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = {}
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Helper Functions
def api_request(endpoint, method="GET", data=None, files=None, require_auth=True):
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    
    if require_auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return response.json(), None
        else:
            return None, response.json().get("detail", "An error occurred")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to server. Please ensure the backend is running."
    except Exception as e:
        return None, str(e)

def get_risk_color(safety_score):
    """Get color based on safety score"""
    if safety_score >= 70:
        return "🟢", "risk-low", "Low Risk"
    elif safety_score >= 40:
        return "🟡", "risk-medium", "Medium Risk"
    else:
        return "🔴", "risk-high", "High Risk"

# Navigation
def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# Pages

def login_page():
    """Login Page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='main-header'>💊 SafeRx</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>AI-Powered Prescription Verification</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form("login_form"):
            st.subheader("🔐 Login")
            email = st.text_input("Email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_btn = st.form_submit_button("Login", use_container_width=True)
            with col_b:
                signup_btn = st.form_submit_button("Sign Up", use_container_width=True)
            
            if login_btn:
                if email and password:
                    with st.spinner("Authenticating..."):
                        result, error = api_request(
                            "/auth/login",
                            method="POST",
                            data={"email": email, "password": password},
                            require_auth=False
                        )
                        
                        if result:
                            st.session_state.logged_in = True
                            st.session_state.token = result["access_token"]
                            st.session_state.user = result["user"]
                            st.success("Login successful!")
                            navigate_to('dashboard')
                        else:
                            st.error(f"Login failed: {error}")
                else:
                    st.warning("Please enter both email and password")
            
            if signup_btn:
                navigate_to('register')

def register_page():
    """Registration Page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='main-header'>📝 Register</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        with st.form("register_form"):
            name = st.text_input("Full Name *")
            email = st.text_input("Email *")
            password = st.text_input("Password *", type="password")
            confirm_password = st.text_input("Confirm Password *", type="password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
            with col_b:
                gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            
            allergies = st.text_area("Known Allergies (Optional)", placeholder="e.g., Penicillin, Sulfa drugs")
            
            col_c, col_d = st.columns(2)
            with col_c:
                register_btn = st.form_submit_button("Register", use_container_width=True)
            with col_d:
                back_btn = st.form_submit_button("Back to Login", use_container_width=True)
            
            if register_btn:
                if not all([name, email, password, confirm_password]):
                    st.error("Please fill in all required fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Creating account..."):
                        result, error = api_request(
                            "/auth/register",
                            method="POST",
                            data={
                                "name": name,
                                "email": email,
                                "password": password,
                                "age": age,
                                "gender": gender,
                                "allergies": allergies
                            },
                            require_auth=False
                        )
                        
                        if result:
                            st.success("Registration successful! Please login.")
                            navigate_to('login')
                        else:
                            st.error(f"Registration failed: {error}")
            
            if back_btn:
                navigate_to('login')

def dashboard_page():
    """Dashboard Page"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.user.get('name', 'User')}!")
        st.markdown("---")
        
        if st.button("🏠 Dashboard", use_container_width=True):
            navigate_to('dashboard')
        
        if st.button("📤 Upload Prescription", use_container_width=True):
            navigate_to('upload')
        
        if st.button("📊 Analysis Results", use_container_width=True):
            if st.session_state.analysis_result:
                navigate_to('results')
            else:
                st.warning("No analysis available. Upload a prescription first.")
        
        if st.button("💬 Chat Assistant", use_container_width=True):
            navigate_to('chat')
        
        if st.button("📜 My History", use_container_width=True):
            navigate_to('history')
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.token = None
            st.session_state.user = {}
            st.session_state.analysis_result = None
            navigate_to('login')
    
    # Main Content
    st.markdown("<div class='main-header'>🏠 Dashboard</div>", unsafe_allow_html=True)
    st.markdown(f"### Welcome back, {st.session_state.user.get('name')}! 👋")
    
    st.markdown("---")
    
    # Quick Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📤 Upload Prescription")
        st.markdown("Analyze a new prescription by uploading an image or entering text")
        if st.button("Start Analysis", key="upload_quick"):
            navigate_to('upload')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💬 Ask AI Assistant")
        st.markdown("Get instant answers to your medication questions")
        if st.button("Open Chat", key="chat_quick"):
            navigate_to('chat')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📜 View History")
        st.markdown("Access your previous prescription analyses")
        if st.button("View Reports", key="history_quick"):
            navigate_to('history')
        st.markdown("</div>", unsafe_allow_html=True)
    
    # System Info
    st.markdown("---")
    st.markdown("### 📊 System Information")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Analyses", "Loading...")
    with col_b:
        st.metric("Last Analysis", "N/A")
    with col_c:
        st.metric("Account Status", "Active ✅")

def upload_page():
    """Upload & Analyze Prescription"""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user.get('name', 'User')}")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')
    
    st.markdown("<div class='main-header'>📄 Upload Prescription</div>", unsafe_allow_html=True)
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "✍️ Enter Text"])
    
    with tab1:
        st.markdown("### Upload Prescription Image")
        st.info("Supported formats: JPG, PNG")
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Prescription", use_container_width=True)
            
            # ✅ Correctly indented new block
            if st.button("🔍 Extract Text from Image", use_container_width=True):
                with st.spinner("Extracting text using OCR..."):
                    # Properly include filename, bytes, and MIME type so FastAPI recognizes the image
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }

                    result, error = api_request(
                        "/analyze/upload",
                        method="POST",
                        files=files
                    )

                    if result:
                        st.success("✅ Text extracted successfully!")
                        extracted_text = result["extracted_text"]
                        st.session_state.extracted_text = extracted_text

                        st.markdown("### Extracted Text:")
                        st.text_area("Edit if needed:", value=extracted_text, height=200, key="extracted_edit")

                        if st.button("📊 Analyze Prescription", use_container_width=True):
                            analyze_prescription(st.session_state.extracted_text)
                    else:
                        st.error(f"Extraction failed: {error}")
    
    with tab2:
        st.markdown("### Enter Prescription Text Manually")
        
        prescription_text = st.text_area(
            "Paste or type prescription here:",
            height=250,
            placeholder="Example:\nDr. Smith\nPatient: John Doe\nAmoxicillin 500mg - Take twice daily\nIbuprofen 400mg - Take as needed for pain"
        )
        
        if st.button("📊 Analyze Prescription", use_container_width=True, key="analyze_manual"):
            if prescription_text.strip():
                analyze_prescription(prescription_text)
            else:
                st.warning("Please enter prescription text")

def analyze_prescription(text):
    """Analyze prescription and display results"""
    with st.spinner("🔬 Analyzing prescription..."):
        result, error = api_request(
            "/analyze/text",
            method="POST",
            data={
                "text": text,
                "user_email": st.session_state.user.get("email")
            }
        )
        
        if result:
            st.session_state.analysis_result = result
            st.success("✅ Analysis complete!")
            navigate_to('results')
        else:
            st.error(f"Analysis failed: {error}")



def results_page():
    """Display Analysis Results"""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user.get('name', 'User')}")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')
        if st.button("📤 New Analysis"):
            navigate_to('upload')
    
    st.markdown("<div class='main-header'>📊 Analysis Results</div>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_result:
        st.warning("No analysis results available. Please upload a prescription first.")
        if st.button("Upload Prescription"):
            navigate_to('upload')
        return
    
    result = st.session_state.analysis_result
    
    # Safety Score
    safety_score = result.get("safety_score", 50)
    emoji, css_class, risk_text = get_risk_color(safety_score)
    
    st.markdown(f"### Overall Safety Assessment: {emoji} {risk_text}")
    st.progress(safety_score / 100)
    st.markdown(f"<h2 style='text-align: center;'>{safety_score}/100</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Extracted Drugs
    st.markdown("### 💊 Extracted Drugs")
    drugs = result.get("drugs", [])
    
    if drugs:
        for drug in drugs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{drug['name']}**")
            with col2:
                confidence = drug.get('confidence', 0) * 100
                st.markdown(f"*Confidence: {confidence:.1f}%*")
    else:
        st.info("No drugs detected in the prescription")
    
    st.markdown("---")
    
    # Drug Interactions
    st.markdown("### ⚠️ Drug Interaction Check")
    interactions = result.get("interactions", [])
    
    if interactions:
        for interaction in interactions:
            severity = interaction.get("severity", "Unknown")
            
            if severity == "High":
                alert_type = "error"
                icon = "🔴"
            elif severity == "Moderate":
                alert_type = "warning"
                icon = "🟡"
            else:
                alert_type = "info"
                icon = "🟢"
            
            with st.expander(f"{icon} {interaction['drug1']} + {interaction['drug2']} - {severity} Risk"):
                st.markdown(f"**Description:** {interaction['description']}")
                st.markdown(f"**Severity:** {severity}")
    else:
        st.success("✅ No significant drug interactions detected")
    
    st.markdown("---")
    
    # Dosage Recommendations
    st.markdown("### 👶 Age-Specific Dosage Recommendations")
    dosage_recs = result.get("dosage_recommendations", [])
    
    if dosage_recs:
        for rec in dosage_recs:
            with st.expander(f"💊 {rec['drug']} - {rec['age_category']}"):
                st.markdown(f"**Recommended Dose:** {rec['recommended_dose']}{rec['unit']}")
                st.markdown(f"**Frequency:** {rec['frequency']}")
                st.markdown(f"**Notes:** {rec['notes']}")
    else:
        st.info("No specific dosage recommendations available")
    
    st.markdown("---")
    
    # Alternative Suggestions
    st.markdown("### 💡 Alternative Medication Suggestions")
    alternatives = result.get("alternatives", [])
    
    if alternatives:
        for alt in alternatives:
            with st.expander(f"🔄 {alt['original_drug']} → {alt['alternative']}"):
                st.markdown(f"**Alternative:** {alt['alternative']}")
                st.markdown(f"**Category:** {alt['category']}")
                st.markdown(f"**Reason:** {alt['reason']}")
    else:
        st.info("No alternative suggestions at this time")
    
    st.markdown("---")
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                report_result, error = api_request(
                    "/report/generate",
                    method="POST",
                    data=result
                )
                
                if report_result:
                    st.success("Report generated! Click below to download.")
                    filename = report_result.get("filename")
                    if filename:
                        st.markdown(f"[Download Report]({API_BASE_URL}/report/download/{filename})")
                else:
                    st.error(f"Failed to generate report: {error}")
    
    with col2:
        if st.button("🔄 New Analysis", use_container_width=True):
            navigate_to('upload')

def chat_page():
    """AI Chat Assistant"""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user.get('name', 'User')}")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')

    # --- HEADER ---
    st.markdown("<div class='main-header'>💬 AI Medical Assistant</div>", unsafe_allow_html=True)
    st.info("Ask questions about medications, interactions, dosages, and more!")

    # --- Initialize chat state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "new_message" not in st.session_state:
        st.session_state.new_message = False

    # --- Example Questions ---
    st.markdown("### 💡 Example Questions:")
    examples = [
        "Is Amoxicillin safe with Ibuprofen?",
        "What is the dosage of Paracetamol for a 5-year-old?",
        "Can I take Aspirin if I have stomach issues?",
        "What are the side effects of Lisinopril?"
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.current_question = example
                st.session_state.new_message = True

    # --- Input Section ---
    user_question = st.text_input("Ask your question:", key="chat_input")

    if st.button("Send", use_container_width=True) or st.session_state.get("new_message", False):
        question = user_question.strip() or st.session_state.current_question.strip()
        if question:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                result, error = api_request(
                    "/chat",
                    method="POST",
                    data={"question": question}
                )

                if result:
                    answer = result.get("answer", "I'm not sure about that.")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Error: {error}"
                    })

            st.session_state.new_message = False
            st.session_state.current_question = ""
            st.rerun()

    # --- Chat Display (clean format) ---
    st.markdown("### 🗨️ Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 AI:** {msg['content']}")
        st.markdown("---")

def history_page():
    """Prescription History"""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user.get('name', 'User')}")
        if st.button("← Back to Dashboard"):
            navigate_to('dashboard')
    
    st.markdown("<div class='main-header'>📜 Prescription History</div>", unsafe_allow_html=True)
    
    with st.spinner("Loading history..."):
        result, error = api_request(f"/history?email={st.session_state.user.get('email')}")
        
        if result:
            history = result.get("history", [])
            
            if history:
                for i, record in enumerate(history):
                    with st.expander(f"Analysis #{i+1} - {record.get('created_at', 'N/A')[:10]}"):
                        st.markdown(f"**Safety Score:** {record.get('safety_score', 'N/A')}/100")
                        
                        drugs = record.get('extracted_drugs', [])
                        if drugs:
                            st.markdown("**Drugs:**")
                            for drug in drugs:
                                st.markdown(f"- {drug.get('name', 'Unknown')}")
                        
                        interactions = record.get('interactions', [])
                        if interactions:
                            st.markdown(f"**Interactions Detected:** {len(interactions)}")
                        
                        st.markdown("**Prescription Text:**")
                        st.text(record.get('prescription_text', 'N/A')[:200] + "...")
            else:
                st.info("No prescription history found. Upload your first prescription to get started!")
        else:
            st.error(f"Failed to load history: {error}")

# Main App Logic
def main():
    if not st.session_state.logged_in:
        if st.session_state.page == 'register':
            register_page()
        else:
            login_page()
    else:
        if st.session_state.page == 'dashboard':
            dashboard_page()
        elif st.session_state.page == 'upload':
            upload_page()
        elif st.session_state.page == 'results':
            results_page()
        elif st.session_state.page == 'chat':
            chat_page()
        elif st.session_state.page == 'history':
            history_page()
        else:
            dashboard_page()

if __name__ == "__main__":
    main()