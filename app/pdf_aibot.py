import streamlit as st
import os
import shutil
import time
from dotenv import load_dotenv
from typing import List
from pathlib import Path

from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda

# Load .env
load_dotenv()

# Streamlit Page Config and Style
st.set_page_config(
    page_title="🤖 RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# === RAG SYSTEM ===
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

        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        os.makedirs(self.temp_pdf_dir, exist_ok=True)

        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            collection_name="markdown_chunks",
            embedding_function=self.embedding_model,
            persist_directory=self.vector_db_dir
        )
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
        self._setup_prompts()
        self.chains = {}

    def _setup_prompts(self):
        self.chat_prompt = PromptTemplate.from_template("""
        You are a helpful assistant. Use the context below to answer the question.
        If the answer is not found, say so clearly.

        Context:
        {context}

        Question:
        {question}
        """)

        self.summarize_prompt = PromptTemplate.from_template("""
        You are an expert summarizer. Your task is to read the provided document context and generate a comprehensive, well-structured summary.

        Context:
        {context}

        Summary:
        """)

        self.quiz_prompt = PromptTemplate.from_template("""
        You are a quiz generator. Create {num_questions} multiple-choice questions (MCQs) from the context.

        Context:
        {context}
        """)

    def parse_pdf(self, uploaded_file) -> List[Document]:
        """Parse a single uploaded PDF using LlamaParse"""
        os.makedirs(self.temp_pdf_dir, exist_ok=True)
        file_path = os.path.join(self.temp_pdf_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
            parsing_instruction=self.parsing_instruction,
        )
        docs = parser.load_data(file_path)
        return docs

    def save_markdown_files(self, docs: List[Document]) -> int:
        saved_count = 0
        for i, doc in enumerate(docs):
            filename = f"document_{i+1}_{int(time.time())}.md"
            file_path = os.path.join(self.markdown_dir, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc.text)
            saved_count += 1
        return saved_count

    def load_and_chunk_documents(self):
        docs = []
        for filename in sorted(os.listdir(self.markdown_dir)):
            if filename.endswith(".md"):
                path = os.path.join(self.markdown_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(Document(page_content=text, metadata={"source": filename}))

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunked = []
        for doc in docs:
            for i, chunk in enumerate(splitter.split_text(doc.page_content)):
                chunked.append(Document(page_content=chunk, metadata={"source": doc.metadata["source"], "chunk": i}))
        return docs, chunked

    def setup_retriever(self, docs: List[Document], chunks: List[Document]):
        self.vectorstore.add_documents(chunks)
        docstore = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=docstore,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        )
        self.retriever.add_documents(docs)
        self._setup_chains()

    def _setup_chains(self):
        parser = StrOutputParser()
        context_retriever = RunnableLambda(lambda x: {"context": self.retriever.get_relevant_documents(x.get("question", "")), **x})

        self.chains = {
            "chat": RunnableSequence(
                context_retriever,
                RunnableLambda(lambda x: {
                    "context": "\n\n".join([doc.page_content for doc in x.get("context", [])]),
                    "question": x.get("question", "")
                }),
                self.chat_prompt,
                self.llm,
                parser
            ),
            "summarize": RunnableSequence(
                context_retriever,
                RunnableLambda(lambda x: {
                    "context": "\n\n".join([doc.page_content for doc in x.get("context", [])])
                }),
                self.summarize_prompt,
                self.llm,
                parser
            ),
            "quiz": RunnableSequence(
                context_retriever,
                RunnableLambda(lambda x: {
                    "context": "\n\n".join([doc.page_content for doc in x.get("context", [])]),
                    "num_questions": x.get("num_questions", 5)
                }),
                self.quiz_prompt,
                self.llm,
                parser
            )
        }

    def process_query(self, mode: str, question: str, num_questions: int = 5):
        if mode not in self.chains:
            return "❌ Invalid mode."
        try:
            data = {"mode": mode, "question": question}
            if mode == "quiz":
                data["num_questions"] = num_questions
            return self.chains[mode].invoke(data)
        except Exception as e:
            return f"❌ Error: {str(e)}"

@st.cache_resource
def init_rag_system():
    return RAGSystem()

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Document Assistant</h1>
        <p>Upload PDFs, ask questions, generate summaries, and create quizzes with AI</p>
    </div>
    """, unsafe_allow_html=True)

    rag = init_rag_system()

    with st.sidebar:
        st.header("📁 Document Management")
        uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'], accept_multiple_files=False)

        if uploaded_file:
            st.info(f"📄 File selected: {uploaded_file.name}")
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing..."):
                    docs = rag.parse_pdf(uploaded_file)
                    if docs:
                        saved = rag.save_markdown_files(docs)
                        markdown_docs, chunks = rag.load_and_chunk_documents()
                        rag.setup_retriever(markdown_docs, chunks)

                        st.session_state.documents_processed = True
                        st.session_state.total_docs = 1
                        st.session_state.total_chunks = len(chunks)

                        st.success(f"✅ Processed 1 document\n📁 Saved {saved} markdown file(s)\n🔍 Created {len(chunks)} chunks")
                    else:
                        st.error("❌ Failed to parse document")

        st.subheader("📊 Document Status")
        if st.session_state.get("documents_processed", False):
            st.success(f"✅ Document parsed")
            st.info(f"📄 {st.session_state.total_chunks} chunks in vector DB")
        else:
            st.warning("⚠️ No document processed yet")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🎯 Query Interface")
        mode = st.selectbox("Select Mode", ["chat", "summarize", "quiz"], format_func=lambda x: {
            "chat": "💬 Chat - Ask questions",
            "summarize": "📋 Summarize - Get summary",
            "quiz": "🧠 Quiz - Generate questions"
        }[x])

        if mode == "chat":
            question = st.text_input("Ask a question:")
        elif mode == "summarize":
            question = "Summarize the document"
            st.info("📋 Summarizing full document.")
        else:
            question = "Generate quiz"
            num_questions = st.slider("Number of questions", 1, 10, 5)

        if st.button("🚀 Process Query", type="primary"):
            if not st.session_state.get("documents_processed", False):
                st.error("❌ Please upload and process a document first.")
            else:
                with st.spinner("Thinking..."):
                    response = rag.process_query(mode, question, num_questions if mode == "quiz" else 5)
                    st.subheader("🤖 Response:")
                    st.markdown(response)

    with col2:
        st.header("ℹ️ Information")
        st.subheader("📊 System Status")
        if os.getenv("LLAMA_CLOUD_API_KEY"):
            st.success("✅ LlamaParse API key found")
        else:
            st.error("❌ Missing LlamaParse API key")

        if os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API key found")
        else:
            st.error("❌ Missing Groq API key")

        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload a PDF** in the sidebar
        2. **Click Process Document**
        3. Choose a mode: Chat, Summarize, or Quiz
        4. Ask a question and click **Process Query**
        """)

if __name__ == "__main__":
    main()
