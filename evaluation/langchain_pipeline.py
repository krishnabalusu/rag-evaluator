"""
LangChain-based RAG pipeline.
Used by RAGAS evaluator — RAGAS integrates natively with LangChain.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from config.settings import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    OPENAI_API_KEY  # we stored GROQ key here
)

WIKI_DOCS_DIR = Path(__file__).parent.parent / "wiki_docs"
EVAL_STORE_DIR = Path(__file__).parent.parent / "eval_vector_store"

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer ONLY based on the provided context.
If the context does not contain the answer, say "I don't have enough information to answer this."
Never hallucinate or make up facts.

Context:
{context}"""


def load_wiki_documents(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Load and chunk Wikipedia articles."""
    loader = DirectoryLoader(
        str(WIKI_DOCS_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} wiki documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def build_vectorstore(chunks: List[Document], collection_name: str = "eval_rag") -> Chroma:
    """Build ChromaDB vector store from chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Clear existing store
    import shutil
    store_path = str(EVAL_STORE_DIR / collection_name)
    if Path(store_path).exists():
        shutil.rmtree(store_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=store_path,
        collection_name=collection_name,
    )
    print(f"Vector store built: {len(chunks)} chunks indexed")
    return vectorstore


def build_rag_chain(vectorstore: Chroma, top_k: int = 5):
    """Build LangChain RAG chain (retriever + LLM)."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


class LangChainRAGPipeline:
    """Full LangChain RAG pipeline — indexing + querying."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, top_k: int = 5):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.chunks = None

    def build(self, collection_name: str = "eval_rag"):
        """Load docs, build vectorstore, build chain."""
        print(f"\n{'='*60}")
        print(f"  Building RAG Pipeline")
        print(f"  chunk_size={self.chunk_size} | overlap={self.chunk_overlap} | top_k={self.top_k}")
        print(f"{'='*60}\n")

        self.chunks = load_wiki_documents(self.chunk_size, self.chunk_overlap)
        self.vectorstore = build_vectorstore(self.chunks, collection_name)
        self.chain, self.retriever = build_rag_chain(self.vectorstore, self.top_k)
        print("\nRAG pipeline ready.\n")
        return self

    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        answer = self.chain.invoke(question)
        retrieved_docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
