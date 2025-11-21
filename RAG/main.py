"""
ALS Support Chatbot API
Provides Swagger documentation, connects to Chroma vector DB,
retrieves context for user queries, and uses LangGraph chatbot for responses.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

from langgraph_chatbot import app as langgraph_app
from rag_setup import load_vectorstore   # Your DB loader
# from rag_chain import clean_text         # Optional if needed

# -------------------------
# Initialize API
# -------------------------
api = FastAPI(
    title="ALS Support Chatbot API",
    description="""
This API powers the ALS Support Chatbot using:

- **RAG Pipeline (ChromaDB) for context retrieval**  
- **LangGraph for conversation flow**  
- **LLM for empathetic answer generation**

You can use `/ask` to ask any ALS-related question.
""",
    version="1.2.0",
    contact={
        "name": "ALS Buddy",
        "url": "https://github.com/Mizbain-Fathima/ALS_Buddy",
    }
)

# -------------------------
# Load Vector DB
# -------------------------
vectorstore = load_vectorstore()


# -------------------------
# Swagger Request + Response Models
# -------------------------

class UserQuery(BaseModel):
    question: str = "What are the early symptoms of ALS?"
    top_k: int = 3


class AnswerResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]


# -------------------------
# Root Route
# -------------------------
@api.get("/", tags=["Health Check"])
def home():
    """
    Simple health-check endpoint.
    """
    return {"message": "ALS Support Chatbot API is running!"}


# -------------------------
# Main Chat Route
# -------------------------
@api.post("/ask", response_model=AnswerResponse, tags=["Chatbot"])
def ask_question(query: UserQuery):
    """
    Ask any ALS-related question.

    **How it works:**
    1. Your query is embedded  
    2. Top `k` similar documents are retrieved from ChromaDB  
    3. RAG context is fed into LangGraph  
    4. LLM produces an empathetic response  
    """

    # 1. Retrieve context from vector DB
    docs = vectorstore.similarity_search(query.question, k=query.top_k)
    contexts = [doc.page_content for doc in docs]
    joined_context = "\n\n".join(contexts)

    # 2. Prepare state for LangGraph
    state = {
        "user_input": query.question,
        "context": joined_context,
        "bot_output": ""
    }

    # 3. Invoke LangGraph chatbot
    result = langgraph_app.invoke(state)

    return AnswerResponse(
        answer=result["bot_output"],
        retrieved_contexts=contexts
    )


# -------------------------
# Uvicorn Server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:api", host="0.0.0.0", port=8000, reload=True)
