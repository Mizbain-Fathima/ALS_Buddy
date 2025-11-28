from fastapi import FastAPI
from pydantic import BaseModel
from RAG.langgraph_chatbot import chat

app = FastAPI()

class Query(BaseModel):
    question: str


@app.post("/chat")
async def ask(query: Query):
    result = chat(query.question)
    return {"answer": result}
