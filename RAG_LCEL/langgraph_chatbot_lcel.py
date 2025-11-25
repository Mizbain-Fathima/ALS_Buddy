"""
High-level orchestration and intent routing. Simplified LCEL-style flow.
"""
from typing import Dict
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer
import torch

from chatbot_with_memory_lcel import chat_with_memory

# Load TinyLlama tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Local TinyLlama model (NO API KEY, NO OPENAI)
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,
    pipeline_kwargs={
        "max_new_tokens": 120,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "return_full_text": False
    }
)


# Intent Classifier using TinyLlama
def classify_intent(text: str) -> str:
    prompt = (
        "Classify the user's intent into one of the following labels:\n"
        "ask_als, personal, out_of_scope.\n\n"
        f"Text: {text}\n\nAnswer with only the label."
    )

    response = llm.invoke(prompt)
    label = response.strip().split()[0].lower()
    return label


# Main message handler
def handle_message(session_id: str, text: str) -> Dict[str, str]:
    intent = classify_intent(text)

    if intent in ("ask_als", "personal"):
        ans = chat_with_memory(session_id, text)
    else:
        ans = (
            "I'm sorry — I don't have the information to help with that topic. "
            "For medical or legal advice, please consult a licensed professional."
        )

    return {"answer": ans, "intent": intent}


# Optional test block
if __name__ == "__main__":
    session = "test_session"
    user_text = input("You: ")
    result = handle_message(session, user_text)
    print("\nBot:", result["answer"])
    print("Intent:", result["intent"])
