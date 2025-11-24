"""
Provides a simple memory layer wrapper around run_rag.
Memory is persisted locally as JSON per session_id.
"""
import json
from pathlib import Path
from typing import Optional
from rag_chain_lcel import run_rag

MEM_DIR = Path(__file__).resolve().parent / 'memory'
MEM_DIR.mkdir(parents=True, exist_ok=True)


def load_memory(session_id: str) -> str:
    path = MEM_DIR / f"{session_id}.json"
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding='utf-8'))
    # assume data is list of {"user":..., "bot":...}
    return "\n".join([f"User: {d['user']}\nBot: {d['bot']}" for d in data])


def update_memory(session_id: str, user_msg: str, bot_msg: str):
    path = MEM_DIR / f"{session_id}.json"
    arr = []
    if path.exists():
        arr = json.loads(path.read_text(encoding='utf-8'))
    arr.append({"user": user_msg, "bot": bot_msg})
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding='utf-8')


def chat_with_memory(session_id: str, user_input: str) -> str:
    mem = load_memory(session_id)
    bot_answer = run_rag(user_input, memory_context=mem)
    update_memory(session_id, user_input, bot_answer)
    return bot_answer
