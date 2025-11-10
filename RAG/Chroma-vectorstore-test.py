# from langchain_community.vectorstores import Chroma
# Because of deprecated warning use langchain_chroma

from langchain_chroma import Chroma

# from langchain_community.embeddings import HuggingFaceEmbeddings
# Because of deprecated warning use langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# Load the persisted Chroma store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Test query
query = "What are the early symptoms of ALS?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content[:500])  # Show first 500 chars
