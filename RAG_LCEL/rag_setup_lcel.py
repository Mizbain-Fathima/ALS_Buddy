"""
RAG chain implemented in LCEL style with NEW Chroma v0.5+ API
"""

from typing import Optional
from pathlib import Path

# NEW LangChain community imports
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

# NEW embeddings 
from langchain_huggingface import HuggingFaceEmbeddings

# NEW Chroma Client
from langchain_chroma import Chroma
from chromadb import PersistentClient

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "gpt-4o-mini"   # change if needed

# RETRIEVER 
def get_retriever(top_k: int = 5):
    """
    Uses NEW Chroma persistent client + LCEL compatible vector store
    """
    client = PersistentClient(path=str(CHROMA_DIR))

    # Access the SAME collection created in rag_setup_lcel.py
    collection = client.get_collection("als_chunks")

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create a LangChain-compatible vectorstore wrapper
    vectorstore = Chroma(
        client=client,
        collection_name="als_chunks",
        embedding_function=embedder,
    )

    return vectorstore.as_retriever(search_kwargs={"k": top_k})

# PROMPT
PROMPT_TPL = """
You are an empathetic assistant for ALS-related questions.
Use the retrieved context to answer concisely, gently, and factually.
Never give medical advice—recommend seeing a doctor when appropriate.

CONTEXT:
{context}

MEMORY:
{memory}

QUESTION:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "memory", "question"],
    template=PROMPT_TPL,
)


# MAIN RAG EXECUTION
def run_rag(
    question: str,
    memory_context: Optional[str] = "",
    top_k: int = 5
) -> str:

    retriever = get_retriever(top_k=top_k)

    # Retrieve docs
    results = retriever.get_relevant_documents(question)
    context = "\n---\n".join([doc.page_content for doc in results]) if results else ""

    # Format prompt
    final_prompt = prompt.format(
        context=context,
        memory=memory_context or "",
        question=question,
    )

    # LLM (LCEL compatible)
    llm = ChatOpenAI(
        temperature=0.2,
        model_name=LLM_MODEL
    )

    response = llm([HumanMessage(content=final_prompt)])

    return response.content
