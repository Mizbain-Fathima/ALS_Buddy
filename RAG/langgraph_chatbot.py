from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain.llms import HuggingFaceHub
from rag_chain import rag_chain
from typing import TypedDict
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# --- Define the state schema ---
class ChatState(TypedDict):
    user_input: str
    context: str
    bot_output: str

# --- Initialize tokenizer & model (TinyLlama Chat) ---
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# --- Initialize model ---
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,
    pipeline_kwargs={
        "max_new_tokens": 280,
        "temperature": 0.4,
        "top_p": 0.9,
        "repetition_penalty": 1.22,
        "no_repeat_ngram_size": 3,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,   # avoids padding warnings
        "return_full_text": False                  # don’t echo the prompt if supported
    }
)

# --- Create LangGraph with schema ---
graph = StateGraph(ChatState)

# --- Node: Retrieve context using RAG ---
def retrieve_state(state: ChatState):
    question = state["user_input"]
    retrieved = rag_chain.invoke({"query": question})
    state["context"] = retrieved["result"]
    return state

# content summarization
def summarize_context(context: str) -> str:
    if len(context.split()) > 250:
        prompt = f"Summarize the following medical context briefly:\n\n{context}\n\nSummary:"
        summary = llm.invoke(prompt)
        return summary.strip()
    return context

def _clean_context(text: str) -> str:
    # Remove prompt-style leftovers that confuse the model
    cleaned = text.replace(
        "Use the following pieces of context to answer the question at the end.", ""
    ).replace(
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.", ""
    )
    return cleaned.strip()

# --- Deduplication helper ---
def _dedupe_lines(text: str) -> str:
    seen, out = set(), []
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out)


# --- Node: Generate context-based answer with empathy ---
# --- Node: Generate context-based answer with empathy ---
def answer_state(state: ChatState):
    context = summarize_context(
    _clean_context(
        _dedupe_lines(state["context"])
        )
    )
    question = state["user_input"]

    # Build a proper chat prompt using the model's chat template
    messages = [
        {"role": "system",
         "content": ("You are an empathetic medical assistant specializing in ALS. "
                     "Answer clearly, concisely (3–5 sentences), and avoid repetition. "
                     "If the context is insufficient, say you’re not sure.")},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # You can also pass stop sequences; TinyLlama usually stops at EOS,
    # but adding role tags as stops can help:
    response = llm.invoke(chat_prompt, stop=["<|system|>", "<|user|>", "<|assistant|>"])
    state["bot_output"] = response.strip()
    return state

# --- Node: Handle additional empathy ---
def empathy_state(state: ChatState):
    user_input = state["user_input"].lower()
    if any(word in user_input for word in ["sad", "worried", "scared", "afraid"]):
        empathy_prompt = """
You are a kind and supportive assistant. Write a short, comforting message to someone feeling anxious about ALS.
"""
        state["bot_output"] += "\n\n" + llm.invoke(empathy_prompt)
    return state

# --- Build Graph ---
graph.add_node("retrieve", retrieve_state)
graph.add_node("answer", answer_state)
graph.add_node("empathy", empathy_state)

graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "empathy")
graph.add_edge("empathy", END)
graph.add_edge("__start__", "retrieve")

app = graph.compile()

# --- Test run ---
if __name__ == "__main__":
    user_input = input("You: ")
    state = {"user_input": user_input, "context": "", "bot_output": ""}
    result = app.invoke(state)
    print("\nBot:", result["bot_output"])

