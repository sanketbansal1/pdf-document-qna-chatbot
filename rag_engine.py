import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self):
        # Initialize HuggingFace Embeddings (local, no API required) and Google GenAI LLM
        # Ensure GOOGLE_API_KEY is in the environment
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
            
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
        
        # We will use an in-memory or temp directory Chroma instance for this example
        # For a production app, you might want a persistent directory
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.prompt = None

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> str:
        """
        Takes PDF bytes, saves them to a temp file, loads, chunks, and indexes them.
        Returns a status message.
        """
        # Save bytes to a temporary file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name

        try:
            # 1. Load the Document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # 2. Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)

            # 3. Create/Update Vector Store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                )
            else:
                self.vector_store.add_documents(chunks)

            # 4. Create Retriever and Chain
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self._setup_chain()

            return f"Successfully processed '{filename}'! Extracted {len(chunks)} chunks of text."

        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _setup_chain(self):
        """Sets up the conversational retrieval chain using modern LangChain API."""
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ])
        self.qa_chain = True

    def ask_question(self, question: str) -> str:
        """Queries the retriever and LLM directly and returns the answer."""
        if not self.retriever or not self.prompt:
            return "Please upload and process a PDF document first before asking questions."

        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        formatted_prompt = self.prompt.format(question=question, context=context)
        response = self.llm.invoke(formatted_prompt)

        # Extract text content from the response
        if hasattr(response, "content"):
            content = response.content
            # If content is a list of dicts (from Google API), extract text
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        return item["text"]
            # If it's a string, return it directly
            if isinstance(content, str):
                return content
        
        return str(response)
