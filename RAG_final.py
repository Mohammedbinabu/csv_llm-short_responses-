import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="CSV RAG", layout="wide")
st.title("CSV Chatbot using RAG")

# Sidebar to upload CSV
st.sidebar.header("📂 Upload CSV")
st.sidebar.markdown("CSV should have `Prompt` and `Response` columns.")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Caching the CSV reading process
@st.cache_data(show_spinner="Reading and caching CSV...")
def read_csv(file):
    return pd.read_csv(file)

# Process CSV file when uploaded
if uploaded_file:
    try:
        # Read the uploaded CSV
        df = read_csv(uploaded_file)
        
        # Ensure the file contains the necessary columns
        if 'Prompt' not in df.columns or 'Response' not in df.columns:
            st.error("The uploaded CSV must contain 'Prompt' and 'Response' columns.")
        else:
            # Keep only the necessary columns and drop rows with missing values
            df = df[['Prompt', 'Response']].dropna()
            
            # Display a preview of the data
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(5))
            
            # Combine 'Prompt' and 'Response' into a single text representation
            df['text_representation'] = df.apply(lambda row: f"Prompt: {row['Prompt']}\nResponse: {row['Response']}", axis=1)
            
            # Convert the text into documents for FAISS vector store
            # text = [Document(page_content=text) for text in df['text_representation']]
            text = [Document(page_content=prompt) for prompt in df['Prompt']]
            db = FAISS.from_documents(text, embedding_model)

            # Initialize chat history if not already present
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # User input field for chat
            user_input = st.chat_input("Ask a question")

            if user_input:
                # Add the user's question to chat history
                st.session_state.chat_history.append(("user", user_input))
                
                # Perform similarity search on the uploaded data
                with st.spinner("Thinking..."):
                    docs = db.similarity_search(user_input)
                    matched_prompt = docs[0].page_content
                    matched_response = df[df['Prompt'] == matched_prompt]['Response'].values[0]
                
                # Add assistant's response to chat history
                st.session_state.chat_history.append(("assistant", matched_response))

            # Display the chat history
            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(msg)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
else:
    st.info("Upload a CSV from the sidebar to begin chatting.")
