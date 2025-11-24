"""
Simple FastAPI server that exposes POST /ask and uses the LCEL workflow above.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langgraph_chatbot_lcel import handle_message

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    session_id: str = 'default_session'

class AskResponse(BaseModel):
    answer: str
    intent: str

@app.post('/ask', response_model=AskResponse)
async def ask(req: AskRequest):
    out = handle_message(req.session_id, req.question)
    return AskResponse(answer=out['answer'], intent=out.get('intent','unknown'))

if __name__ == '__main__':
    uvicorn.run('api_main_lcel:app', host='0.0.0.0', port=8080, reload=True)
