import streamlit as st
import pandas as pd
import os
import io
import requests
from classify import classify, classify_log
from processor_regex import classify_with_regex

# Set page config
st.set_page_config(
    page_title="AI Log Classifier Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, high-contrast light theme design
st.markdown("""
<style>
    /* Clean light background & main text styling */
    .stApp {
        background: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Force all text elements to have high contrast */
    p, span, label, div {
        color: #1e293b !important;
    }
    
    /* Header Card */
    .main-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a !important;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .sub-title {
        color: #475569 !important;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
        color: #334155 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb !important;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Tab navigation text contrast */
    button[data-baseweb="tab"] p {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    button[aria-selected="true"] p {
        color: #2563eb !important;
    }

    /* Result Badges */
    .badge-security { background-color: #dc2626; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-workflow { background-color: #d97706; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-system { background-color: #2563eb; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-deprecation { background-color: #7c3aed; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-http { background-color: #059669; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }
    .badge-user { background-color: #0891b2; color: white !important; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

    /* Custom Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    .stButton>button p {
        color: #ffffff !important;
    }

    .stButton>button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=75)
    else:
        st.markdown("<h1 style='margin:0;'>⚡</h1>", unsafe_allow_html=True)
    st.title("Log Classifier")
    st.caption("Hybrid ML & LLM Log Classification Engine")
    
    st.divider()
    st.markdown("### 🧩 Pipeline Architecture")
    st.markdown("""
    - **LegacyCRM**: Groq LLM Classifier
    - **Standard Sources**:
      1. Regex Pattern Matching
      2. SentenceTransformer (`MiniLM-L6-v2`) + BERT Classifier
    """)
    
    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- **FastAPI & Uvicorn**\n- **Streamlit Dashboard**\n- **SentenceTransformers & Scikit-Learn**\n- **Groq LLM API**\n- **Pandas & Joblib**")
    
    st.divider()
    st.caption("Developed for Production Log Analysis")

# Main Header
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ Log Classification Intelligence Platform</div>
    <div class="sub-title">Automated real-time categorization of system, security, workflow, and API logs using hybrid Regex, BERT Embedding, and Groq LLM models.</div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📁 Batch CSV Classifier", "⚡ Real-Time Log Tester", "📡 API Endpoint & Architecture"])

# ==========================================
# TAB 1: BATCH CSV CLASSIFIER
# ==========================================
with tab1:
    st.markdown("### Batch Log Classification")
    st.caption("Upload a CSV file containing `source` and `log_message` columns or test with our sample dataset.")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Choose a CSV log file", type=["csv"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📥 Load Sample CSV (`test.csv`)", use_container_width=True)

    df_input = None

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
    elif use_sample:
        sample_path = "resources/test.csv"
        if os.path.exists(sample_path):
            df_input = pd.read_csv(sample_path)
            st.success("Sample CSV loaded successfully!")
        else:
            st.warning(f"Sample file at `{sample_path}` not found.")

    if df_input is not None:
        if "source" not in df_input.columns or "log_message" not in df_input.columns:
            st.error("Uploaded CSV must contain `source` and `log_message` columns.")
        else:
            with st.spinner("Classifying log entries..."):
                logs_tuple = list(zip(df_input["source"], df_input["log_message"]))
                df_input["target_label"] = classify(logs_tuple)

            st.markdown("---")
            st.markdown("### 📈 Classification Summary")

            # Metrics
            total_logs = len(df_input)
            counts = df_input["target_label"].value_counts().to_dict()

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Logs", total_logs)
            m2.metric("Security Alerts", counts.get("Security Alert", 0))
            m3.metric("System Notifications", counts.get("System Notification", 0))
            m4.metric("Workflow Errors", counts.get("Workflow Error", 0))
            m5.metric("Deprecation Warnings", counts.get("Deprecation Warning", 0))

            st.markdown("<br>", unsafe_allow_html=True)

            # Interactive View and Download
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown("#### Classified Logs Table")
                
                # Category Filter
                all_cats = ["All"] + list(df_input["target_label"].unique())
                selected_cat = st.selectbox("Filter by Category", all_cats)
                
                filtered_df = df_input if selected_cat == "All" else df_input[df_input["target_label"] == selected_cat]
                st.dataframe(filtered_df, use_container_width=True, height=350)
                
                # Export Button
                csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Classified Results CSV",
                    data=csv_bytes,
                    file_name="classified_logs_output.csv",
                    mime="text/csv"
                )

            with col_right:
                st.markdown("#### Category Distribution")
                st.bar_chart(df_input["target_label"].value_counts())

# ==========================================
# TAB 2: REAL-TIME SINGLE LOG TESTER
# ==========================================
with tab2:
    st.markdown("### Real-Time Single Log Classification")
    st.caption("Test individual log lines dynamically to see which classification model handles them.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        source_option = st.selectbox(
            "Log Source",
            ["LegacyCRM", "ModernCRM", "BillingSystem", "AnalyticsEngine", "ModernHR", "CustomSource"]
        )
        st.info(
            "ℹ️ **Routing Logic**:\n"
            "- **`LegacyCRM`** -> Evaluated via **Groq LLM**\n"
            "- **Others** -> Evaluated via **Regex**, then **SentenceTransformer + BERT** fallback"
        )

    with col_s2:
        log_input = st.text_area(
            "Log Message",
            height=120,
            value="Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active." if source_option == "LegacyCRM" else "IP 192.168.133.114 blocked due to potential attack"
        )
        
        classify_btn = st.button("🚀 Classify Log Message", use_container_width=True)

    if classify_btn and log_input.strip():
        with st.spinner("Analyzing log message..."):
            # Determine engine used for transparency
            if source_option == "LegacyCRM":
                engine_used = "Groq LLM API (`openai/gpt-oss-20b`)"
            else:
                regex_res = classify_with_regex(log_input)
                if regex_res:
                    engine_used = "Regex Pattern Engine"
                else:
                    engine_used = "SentenceTransformer (`MiniLM-L6-v2`) + BERT Classifier"
            
            result_label = classify_log(source_option, log_input)

        st.markdown("---")
        st.markdown("### 🎯 Classification Result")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(f"**Predicted Label:**")
            st.subheader(f"🏷️ {result_label}")
        with res_col2:
            st.markdown(f"**Inference Engine Used:**")
            st.info(f"⚙️ {engine_used}")

# ==========================================
# TAB 3: API ENDPOINT & ARCHITECTURE
# ==========================================
with tab3:
    st.markdown("### REST API Endpoint & Integration")
    st.markdown("""
    The system exposes a high-performance **FastAPI** REST API endpoint at `/classify/`.
    
    #### 📡 API Endpoint Details:
    - **URL**: `http://127.0.0.1:8000/classify/`
    - **HTTP Method**: `POST`
    - **Payload**: `multipart/form-data` with a `file` field containing a CSV (`source`, `log_message`).
    - **Response**: CSV File download with added `target_label` column.
    """)

    st.divider()

    st.markdown("### 💻 Code Snippet to Test Endpoint")
    st.code("""
import requests

url = "http://127.0.0.1:8000/classify/"
files = {'file': ('logs.csv', open('resources/test.csv', 'rb'), 'text/csv')}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('classified_output.csv', 'wb') as f:
        f.write(response.content)
    print("Classification complete!")
else:
    print("Error:", response.status_code, response.text)
    """, language="python")

    st.divider()
    
    st.markdown("### 📊 System Architecture Diagram")
    st.markdown("""
    ```mermaid
    flowchart TD
        A[Input Log CSV / Message] --> B{Check Source}
        B -- "source == LegacyCRM" --> C[Processor LLM: Groq API]
        B -- "source != LegacyCRM" --> D[Processor Regex: Pattern Matching]
        D -- "Match Found" --> E[Return Regex Label]
        D -- "No Match" --> F[Processor BERT: SentenceTransformer Embeddings]
        F --> G[Joblib Classifier: Logistic Regression]
        C --> H[Target Label Output]
        E --> H
        G --> H
    ```
    """)
