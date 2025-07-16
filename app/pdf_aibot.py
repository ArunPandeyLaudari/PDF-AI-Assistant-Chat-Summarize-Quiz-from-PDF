import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time

# Core dependencies
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# LangChain dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableBranch

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #007bff;
        margin-bottom: 20px;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .mode-selector {
        background: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

class RAGSystem:
    def __init__(self):
        self.parsing_instruction = """
        You are a document parser. Extract the content into clean, structured markdown format.
        - Preserve headings, subheadings, paragraphs clearly.
        - Convert tables into proper markdown table syntax.
        - Represent images with markdown image syntax ![Description](image_placeholder).
        - If image data is missing, describe the image briefly in place.
        - Keep lists, bullet points, and code blocks formatted.
        - Avoid extra line breaks or broken markdown syntax.
        """
        
        self.markdown_dir = "markdown_files"
        self.vector_db_dir = "chroma_vectorstore"
        self.temp_pdf_dir = "temp_pdfs"
        
        # Create directories
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        os.makedirs(self.temp_pdf_dir, exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chains = {}
        
        self._setup_models()
        self._setup_prompts()
    
    def _setup_models(self):
        """Initialize embedding model and LLM"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            
            self.llm = ChatGroq(
                model_name="llama-3.1-8b-instant", 
                temperature=0.2
            )
            
            # Initialize vectorstore
            self.vectorstore = Chroma(
                collection_name="markdown_chunks",
                embedding_function=self.embedding_model,
                persist_directory=self.vector_db_dir
            )
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        self.chat_prompt = PromptTemplate.from_template("""
        You are a helpful assistant. Use the context below to answer the question.
        If the answer is not found, say so clearly.

        Context:
        {context}

        Question:
        {question}
        """)

        self.summarize_prompt = PromptTemplate.from_template("""
        You are an expert summarizer. Your task is to read the provided document context and generate a comprehensive, well-structured summary that captures all key points, main ideas, and important details.

        Instructions:
        - Organize the summary with clear headings and bullet points where appropriate.
        - Highlight major sections, concepts, and any lists or processes described in the context.
        - Use concise language, but do not omit critical information.
        - If the context includes tables, describe their content in summary form.
        - If there are images or diagrams referenced, briefly mention their purpose or content.
        - The summary should be easy to read and suitable for someone who needs a quick but thorough understanding of the document.

        Context:
        {context}

        Summary:
        """)

        self.quiz_prompt = PromptTemplate.from_template("""
        You are a quiz generator. Create {num_questions} multiple-choice questions (MCQs) from the context below.
        
        Format each question as:
        **Question X:** [Question text]
        A) [Option A] \n
        B) [Option B] \n
        C) [Option C] \n
        D) [Option D] \n 
        
        **Correct Answer:** [Letter]
        
        Context:
        {context}
        """)
    
    def parse_pdfs(self, uploaded_files) -> List[Document]:
        """Parse uploaded PDF files using LlamaParse"""
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            st.error("LLAMA_CLOUD_API_KEY not found in environment variables!")
            return []
        
        # Clear temp directory
        shutil.rmtree(self.temp_pdf_dir, ignore_errors=True)
        os.makedirs(self.temp_pdf_dir, exist_ok=True)
        
        # Save uploaded files
        for uploaded_file in uploaded_files:
            with open(os.path.join(self.temp_pdf_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        try:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",
                verbose=True,
                parsing_instruction=self.parsing_instruction,
            )

            loader = SimpleDirectoryReader(
                input_dir=self.temp_pdf_dir,
                file_extractor={".pdf": parser}
            )

            docs = loader.load_data()
            return docs
            
        except Exception as e:
            st.error(f"Error parsing PDFs: {str(e)}")
            return []
    
    def save_markdown_files(self, docs: List[Document]) -> int:
        """Save parsed documents as markdown files"""
        saved_count = 0
        
        for i, doc in enumerate(docs):
            filename = f"document_{i+1}_{int(time.time())}.md"
            file_path = os.path.join(self.markdown_dir, filename)
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(doc.text)
                saved_count += 1
            except Exception as e:
                st.error(f"Error saving {filename}: {str(e)}")
        
        return saved_count
    
    def load_and_chunk_documents(self) -> List[Document]:
        """Load markdown files and create chunks"""
        markdown_docs = []
        
        # Load all markdown files
        for filename in sorted(os.listdir(self.markdown_dir)):
            if filename.endswith(".md"):
                path = os.path.join(self.markdown_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    markdown_docs.append(Document(page_content=text, metadata={"source": filename}))
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
        
        # Create chunks
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunked_docs = []
        
        for doc in markdown_docs:
            splits = child_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(splits):
                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": doc.metadata["source"], "chunk": i}
                    )
                )
        
        return markdown_docs, chunked_docs
    
    def setup_retriever(self, markdown_docs: List[Document], chunked_docs: List[Document]):
        """Setup the parent document retriever"""
        try:
            # Add chunks to vectorstore
            if chunked_docs:
                self.vectorstore.add_documents(chunked_docs)
            
            # Setup parent document retriever
            docstore = InMemoryStore()
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter
            )
            
            if markdown_docs:
                self.retriever.add_documents(markdown_docs)
            
            self._setup_chains()
            
        except Exception as e:
            st.error(f"Error setting up retriever: {str(e)}")
    
    def _setup_chains(self):
        """Setup the processing chains"""
        parser = StrOutputParser()
        
        # Context retriever
        context_retriever = RunnableLambda(
            lambda x: {"context": self.retriever.get_relevant_documents(x.get("question", "")), **x}
        )
        
        # Formatters
        chat_formatter = RunnableLambda(lambda x: {
            "context": "\n\n".join([doc.page_content for doc in x.get("context", [])]),
            "question": x.get("question", "")
        })

        summarize_formatter = RunnableLambda(lambda x: {
            "context": "\n\n".join([doc.page_content for doc in x.get("context", [])])
        })

        quiz_formatter = RunnableLambda(lambda x: {
            "context": "\n\n".join([doc.page_content for doc in x.get("context", [])]),
            "num_questions": x.get("num_questions", 5)
        })
        
        # Create chains
        self.chains = {
            "chat": RunnableSequence(
                context_retriever,
                chat_formatter,
                self.chat_prompt,
                self.llm,
                parser
            ),
            "summarize": RunnableSequence(
                context_retriever,
                summarize_formatter,
                self.summarize_prompt,
                self.llm,
                parser
            ),
            "quiz": RunnableSequence(
                context_retriever,
                quiz_formatter,
                self.quiz_prompt,
                self.llm,
                parser
            )
        }
    
    def process_query(self, mode: str, question: str, num_questions: int = 5) -> str:
        """Process user query based on mode"""
        if mode not in self.chains:
            return "❌ Invalid mode selected. Choose 'chat', 'summarize', or 'quiz'."
        
        try:
            input_data = {
                "mode": mode,
                "question": question
            }
            
            if mode == "quiz":
                input_data["num_questions"] = num_questions
            
            response = self.chains[mode].invoke(input_data)
            return response
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

# Initialize RAG system
@st.cache_resource
def init_rag_system():
    return RAGSystem()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Document Assistant</h1>
        <p>Upload PDFs, ask questions, generate summaries, and create quizzes with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = init_rag_system()
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        st.subheader("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
            
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Parse PDFs
                    docs = rag_system.parse_pdfs(uploaded_files)
                    
                    if docs:
                        # Save markdown files
                        saved_count = rag_system.save_markdown_files(docs)
                        
                        # Load and chunk documents
                        markdown_docs, chunked_docs = rag_system.load_and_chunk_documents()
                        
                        # Setup retriever
                        rag_system.setup_retriever(markdown_docs, chunked_docs)
                        
                        st.success(f"✅ Processed {len(docs)} documents\n📁 Saved {saved_count} markdown files\n🔍 Created {len(chunked_docs)} chunks")
                        
                        # Store in session state
                        st.session_state.documents_processed = True
                        st.session_state.total_docs = len(docs)
                        st.session_state.total_chunks = len(chunked_docs)
                    else:
                        st.error("❌ Failed to process documents")
        
        # Document status
        st.subheader("📊 Document Status")
        if hasattr(st.session_state, 'documents_processed') and st.session_state.documents_processed:
            st.success(f"✅ {st.session_state.total_docs} documents processed")
            st.info(f"📄 {st.session_state.total_chunks} chunks in vector store")
        else:
            st.warning("⚠️ No documents processed yet")
        
        # View saved files
        if st.button("📂 View Saved Files"):
            markdown_files = [f for f in os.listdir(rag_system.markdown_dir) if f.endswith('.md')]
            if markdown_files:
                st.subheader("Saved Markdown Files:")
                for file in markdown_files:
                    st.text(f"📄 {file}")
            else:
                st.info("No markdown files found")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Query Interface")
        
        # Mode selection
       
        mode = st.selectbox(
            "Select Mode",
            ["chat", "summarize", "quiz"],
            format_func=lambda x: {
                "chat": "💬 Chat - Ask questions about documents",
                "summarize": "📋 Summarize - Get document summaries",
                "quiz": "🧠 Quiz - Generate quiz questions"
            }[x]
        )
        
        
        # Input fields based on mode
        if mode == "chat":
            question = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is the main topic of the document?"
            )
        elif mode == "summarize":
            question = "Summarize the document"
            st.info("📋 Summarize mode selected - will generate a comprehensive summary")
        elif mode == "quiz":
            question = "Generate quiz"
            num_questions = st.slider("Number of questions:", 1, 10, 5)
            st.info(f"🧠 Quiz mode selected - will generate {num_questions} questions")
        
        # Process query
        if st.button("🚀 Process Query", type="primary"):
            if not hasattr(st.session_state, 'documents_processed') or not st.session_state.documents_processed:
                st.error("❌ Please upload and process documents first!")
            else:
                with st.spinner("Processing your query..."):
                    if mode == "quiz":
                        response = rag_system.process_query(mode, question, num_questions)
                    else:
                        response = rag_system.process_query(mode, question)
                    
                    # Display response
                    st.subheader("🤖 Response:")
                    st.markdown(response)
    
    with col2:
        st.header("ℹ️ Information")
        
        # Quick stats
        st.subheader("📊 System Status")
        
        # Environment check
        if os.getenv("LLAMA_CLOUD_API_KEY"):
            st.success("✅ LlamaParse API Key found")
        else:
            st.error("❌ LlamaParse API Key missing")
        
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API Key found")
        else:
            st.error("❌ Groq API Key missing")
        
        
        
        # Usage instructions
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload PDFs** in the sidebar
        2. **Process Documents** to parse and store them
        3. **Select a Mode**:
           - 💬 **Chat**: Ask specific questions
           - 📋 **Summarize**: Get document summaries
           - 🧠 **Quiz**: Generate quiz questions
        4. **Process Query** to get AI responses
        """)
        
        
        # Features
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("🚀 Features")
        st.markdown("""
        - 📄 **PDF Parsing** with LlamaParse
        - 🔍 **Vector Search** with Chroma
        - 💾 **Auto-save** markdown files
        - 🧠 **Parent-Child Retrieval**
        - 💬 **Multi-mode** interactions
        - 🎨 **Professional UI**
        """)
        

if __name__ == "__main__":
    main()