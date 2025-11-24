"""
Usage: import rag_chain_lcel and call `run_rag(question, memory_context=None)`
This file demonstrates LCEL-style composition using LangChain primitives.
"""
from typing import Optional, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / 'chroma_db'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'gpt-4o-mini'  # replace with available model or use HF model wrapper

# Retriever
def get_retriever(top_k: int = 5):
    client = chromadb.Client(Settings(persist_directory=str(CHROMA_DIR), chroma_db_impl="duckdb+parquet"))
    collection = client.get_collection(name='als_chunks')
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    chroma = Chroma(collection_name='als_chunks', embedding_function=embed, client=client)
    return chroma.as_retriever(search_kwargs={"k": top_k})

# Prompt template
PROMPT_TPL = """
You are an empathetic assistant for ALS-related questions. Use the
context below to answer the user's question. Be concise, factual,
and gently worded. Do not provide medical advice — recommend seeking
professional help when appropriate.

CONTEXT:
{context}

MEMORY:
{memory}

USER QUESTION:
{question}
"""

prompt = PromptTemplate(input_variables=["context", "memory", "question"], template=PROMPT_TPL)


def run_rag(question: str, memory_context: Optional[str] = "", top_k: int = 5) -> str:
    retriever = get_retriever(top_k=top_k)
    results = retriever.get_relevant_documents(question)
    context = "\n---\n".join([r.page_content for r in results]) if results else ""

    final_prompt = prompt.format(context=context, memory=memory_context or "", question=question)

    # Use a chat model — you can swap to HF models
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    resp = llm([HumanMessage(content=final_prompt)])
    return resp.content
