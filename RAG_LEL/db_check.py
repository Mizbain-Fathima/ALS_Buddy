import os
from RAG_LEL.rag_setup import PERSIST_DIR, get_retriever

print("DB PATH:", PERSIST_DIR)
print("EXISTS:", os.path.exists(PERSIST_DIR))

retriever = get_retriever()
docs = retriever.invoke("ALS")

print("DOC COUNT:", len(docs))

if docs:
    print("\n--- SAMPLE DOCUMENT ---")
    print(docs[0].page_content[:500])
