import streamlit as st
import requests
import json
import time
from datetime import datetime
import os
from typing import Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ResearchAssistantUI:
    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        if 'current_research' not in st.session_state:
            st.session_state.current_research = None
        if 'research_plan' not in st.session_state:
            st.session_state.research_plan = None
        if 'show_advanced' not in st.session_state:
            st.session_state.show_advanced = False
    
    def render_sidebar(self):
        """Render sidebar with settings and history"""
        with st.sidebar:
            st.title("🔬 Research Assistant")
            st.markdown("---")
            
            # API Status and Data Status
            self.check_api_status()
            self.check_data_status()
            
            st.markdown("---")
            
            # Research History
            st.subheader("📚 Research History")
            if st.session_state.research_history:
                for i, research in enumerate(reversed(st.session_state.research_history)):
                    with st.expander(f"{research['topic'][:30]}..."):
                        st.write(f"**Date:** {research['date']}")
                        st.write(f"**Quality:** {research.get('quality_score', 'N/A')}")
                        st.write(f"**Sources:** {research.get('source_count', 'N/A')}")
                        if st.button(f"Rerun Research", key=f"rerun_{i}"):
                            st.session_state.current_topic = research['topic']
                            st.rerun()
            else:
                st.info("No research history yet")
            
            st.markdown("---")
            
            # Settings
            st.subheader("⚙️ Settings")
            st.session_state.show_advanced = st.checkbox("Show Advanced Options")
            
            # Clear history
            if st.button("🗑️ Clear History"):
                st.session_state.research_history = []
                st.rerun()
    
    def check_api_status(self):
        """Check and display API status"""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except requests.exceptions.RequestException:
            st.error("❌ API Disconnected")
    
    def check_data_status(self):
        """Check and display data status"""
        try:
            response = requests.get(f"{self.api_base_url}/data/status", timeout=5)
            if response.status_code == 200:
                data_status = response.json()
                
                st.subheader("📊 Data Status")
                st.metric("PDF Files", data_status.get('pdf_files_found', 0))
                st.metric("Documents Loaded", data_status.get('documents_in_collection', 0))
                
                if data_status.get('is_loaded', False):
                    st.success("✅ Data Loaded")
                else:
                    st.warning("⚠️ No Data Loaded")
                    
                if st.button("🔄 Reload Data"):
                    self.reload_data()
            else:
                st.error("❌ Data Status Error")
        except requests.exceptions.RequestException:
            st.error("❌ Cannot Check Data Status")
    
    def reload_data(self):
        """Reload data from data folder"""
        try:
            with st.spinner("🔄 Reloading data..."):
                response = requests.post(f"{self.api_base_url}/data/reload", timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    if result["status"] == "completed":
                        st.success(f"✅ Reloaded {result['successful']} files")
                    else:
                        st.warning(f"⚠️ {result.get('message', 'Unknown result')}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to reload data: {response.text}")
        except Exception as e:
            st.error(f"❌ Error reloading data: {str(e)}")
    
    def render_main_interface(self):
        """Render main research interface"""
        st.title("🔬 Agentic Research Assistant")
        st.markdown("### Intelligent Multi-Agent Research System")
        
        # Data folder info
        st.info("📁 This system uses PDF files from your `data/documents/` folder. Add your research documents there and reload data if needed.")
        
        # Research input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input(
                "Research Topic",
                placeholder="Enter your research topic here...",
                help="Describe what you want to research. Be specific for better results."
            )
        
        with col2:
            research_type = st.selectbox(
                "Research Type",
                ["Comprehensive", "Quick Overview", "Deep Analysis"]
            )
        
        # Advanced options
        if st.session_state.show_advanced:
            with st.expander("🔧 Advanced Options"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quality_threshold = st.slider(
                        "Quality Threshold", 
                        0.0, 1.0, 0.7, 0.1,
                        help="Minimum quality score before generating report"
                    )
                
                with col2:
                    max_iterations = st.slider(
                        "Max Iterations", 
                        1, 5, 3,
                        help="Maximum research iterations for quality improvement"
                    )
                
                with col3:
                    source_preference = st.selectbox(
                        "Source Preference",
                        ["Balanced", "Internal Focus", "External Focus"]
                    )
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📋 Generate Plan", type="secondary", use_container_width=True):
                if topic:
                    self.generate_research_plan(topic)
                else:
                    st.error("Please enter a research topic")
        
        with col2:
            if st.button("🔍 Start Research", type="primary", use_container_width=True):
                if topic:
                    self.start_research(topic, research_type)
                else:
                    st.error("Please enter a research topic")
    
    # ... (keep all the existing methods like generate_research_plan, start_research, etc.)
    # Just remove any upload-related methods
    
    def run(self):
        """Main application runner"""
        self.render_sidebar()
        
        # Main content tabs - removed upload tab
        tab1, tab2 = st.tabs(["🔍 Research", "📈 Analytics"])
        
        with tab1:
            self.render_main_interface()
        
        with tab2:
            self.render_analytics_dashboard()
    
    # ... (include all other existing methods except upload-related ones)

def main():
    """Main function"""
    app = ResearchAssistantUI()
    app.run()

if __name__ == "__main__":
    main()
