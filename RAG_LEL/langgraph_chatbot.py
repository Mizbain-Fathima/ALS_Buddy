from langgraph.graph import StateGraph
from typing import TypedDict
from RAG.rag_chain import answer_question


class ChatState(TypedDict):
    user_input: str
    response: str


# -------- Node Function --------
def rag_node(state: ChatState):
    answer = answer_question(state["user_input"])
    return {"response": answer}


# -------- LangGraph Setup --------
graph = StateGraph(ChatState)
graph.add_node("rag_response", rag_node)
graph.set_entry_point("rag_response")
graph.set_finish_point("rag_response")

chat_graph = graph.compile()


def chat(query: str):
    result = chat_graph.invoke({"user_input": query})
    return result["response"]
