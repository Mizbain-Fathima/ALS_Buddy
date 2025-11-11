# RAG/api_main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_chatbot import app as chatbot_app
from typing import Dict
import uvicorn

# --- Define FastAPI instance ---
api = FastAPI(
    title="ALS Support Chatbot API",
    description="An empathetic chatbot that provides information and emotional support about ALS using RAG + LangGraph.",
    version="1.0"
)

# --- Define request body model ---
class UserQuery(BaseModel):
    question: str

# --- Root Route ---
@api.get("/")
def home():
    return {"message": "Welcome to ALS Support Chatbot API. Use /ask to send a query."}

# --- Chat Endpoint ---
@api.post("/ask")
def ask_question(query: UserQuery) -> Dict[str, str]:
    """
    Takes user input and returns chatbot response.
    """
    state = {"user_input": query.question, "context": "", "bot_output": ""}
    result = chatbot_app.invoke(state)
    return {"answer": result["bot_output"]}

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:api", host="0.0.0.0", port=8000)