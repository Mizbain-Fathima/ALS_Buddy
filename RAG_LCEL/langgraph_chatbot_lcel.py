"""
High-level orchestration and intent routing. Simplified LCEL-style flow.
"""
from typing import Dict
from langchain.chat_models import ChatOpenAI

from chatbot_with_memory_lcel import chat_with_memory

llm = ChatOpenAI(temperature=0)


def classify_intent(text: str) -> str:
    # Simple intent classifier using LLM; you can replace with a lightweight rulebased classifier
    prompt = f"Classify intent as one of: ask_als, personal, out_of_scope.\nText: {text}\nAnswer with single label."
    resp = llm.call_as_llm([{"role":"user","content":prompt}])
    label = resp if isinstance(resp, str) else getattr(resp, 'content', str(resp))
    return label.strip().split()[0]


def handle_message(session_id: str, text: str) -> Dict[str,str]:
    intent = classify_intent(text)
    if intent in ("ask_als", "personal"):
        ans = chat_with_memory(session_id, text)
    else:
        ans = "I'm sorry — I don't have the information to help with that topic. For medical or legal advice, please consult a licensed professional."
    return {"answer": ans, "intent": intent}
