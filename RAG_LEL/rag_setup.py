import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PERSIST_DIR = os.path.abspath(r"D:\ALS-chatbot\RAG\chroma_db")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_retriever():
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings()
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})
