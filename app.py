import streamlit as st
from rag_engine import RAGEngine

# Page config must be the first Streamlit command
st.set_page_config(page_title="PDF Chatbot (RAG)", page_icon="📚", layout="wide")

@st.cache_resource
def get_rag_engine():
    """Caches the RAG Engine so we don't re-initialize on every rerun"""
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        st.stop()

def main():
    st.title("📚 Chat with your PDF Documents")
    st.write("Upload a PDF document and ask questions about its content.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Get the RAG engine
    rag = get_rag_engine()

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing PDF (extracting, chunking, embedding)..."):
                    try:
                        # Read the file as bytes
                        file_bytes = uploaded_file.read()
                        result_msg = rag.process_pdf(file_bytes, uploaded_file.name)
                        st.success(result_msg)
                    except Exception as e:
                        st.error(f"Error processing document: {e}")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = rag.ask_question(prompt)
                    message_placeholder.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating answer: {e}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
