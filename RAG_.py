import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from dotenv import load_dotenv
import tempfile

load_dotenv()
groq_api_key = os.getenv('Grok_API_KEY')
 
st.set_page_config(page_title="CSV Chat with Groq", layout="wide")
st.title("CSV Chatbot with LLaMA 3 (Groq)")
 
# Sidebar for CSV upload
st.sidebar.header("📂 Upload CSV")
st.sidebar.markdown("CSV should have `Prompt` and `Response` columns.")
uploaded_file = st.sidebar.file_uploader("Upload your car CSV file", type=["csv"])
 
# Cache CSV read
@st.cache_data(show_spinner="Reading and caching CSV...")
def read_csv(file):
    return pd.read_csv(file)
 
# Cache database creation
@st.cache_resource(show_spinner="Creating SQL DB...")
def create_sqlite_db(df: pd.DataFrame):
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "car.db")
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("driving_car", engine, index=False, if_exists="replace")
    return SQLDatabase(engine), temp_dir
 
if uploaded_file:
    try:
        df = read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
 
    db, tmp_dir = create_sqlite_db(df)
 
    llm = ChatGroq(model_name="llama3-70b-8192", api_key=groq_api_key)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=False)
 
    # Chat Interface
    st.subheader("💬 Ask me anything about your data")
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
 
    user_input = st.chat_input("Ask a question")
 
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": user_input})
        st.session_state.chat_history.append(("assistant", response.get("output", "No response.")))
 
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
else:
    st.info("Upload a CSV from the sidebar to begin chatting.")