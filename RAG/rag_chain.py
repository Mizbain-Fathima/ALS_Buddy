from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline  # Free, local LLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import transformers
import torch

# Load the persisted vectorstore (from your rag_setup.py)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize a free, local LLM (HuggingFace model)
# Using DistilGPT-2 for speed/efficiency; change to "gpt2" for full GPT-2 if needed
llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",  # Lightweight model; runs locally
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available, else CPU
    model_kwargs={"temperature": 0.4, "max_length": 512}  # Adjust for creativity/response length
)

# Set up retriever and RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True  # Optional: Returns source chunks for transparency
)

# Example query
query = "What are early symptoms of ALS?"
response = rag_chain.invoke({"query": query})  # Updated method for newer LangChain

print("Query:", query)
print("Response:", response["result"])
if "source_documents" in response:
    print("Sources:", [doc.page_content[:200] + "..." for doc in response["source_documents"]])  # Preview sources


