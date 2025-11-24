#from langchain.embeddings import OpenAIEmbeddings # install langchain-openai
#from langchain_openai import OpenAIEmbeddings # install langchain-openai

#Because openai credits are finished, we will use HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Deprecated warning is an harmless warning so in future if needed then use
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

#from langchain.vectorstores import Chroma # install langchain-chroma and for chromdb install chromadb 
from langchain_community.vectorstores import Chroma # install langchain-community and for chromdb install chromadb

#from langchain.docstore.document import Document # install langchain-community
from langchain_core.documents import Document # install langchain-core
import json
from dotenv import load_dotenv
import os

# for HuggingFace embeddings install sentence-transformers 
# and use from langchain_community.embeddings import HuggingFaceEmbeddings

# Load chunks from preprocessing (assuming you saved them or have them in memory)
# If not, load from your JSON or rerun preprocessing
with open('../webscrapped-data/als_articles_expanded.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

all_text = " ".join([d["content"] for d in data])

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_text(all_text)

# Create Document objects
docs = [Document(page_content=text) for text in chunks]

# Initialize embeddings (replace with your OpenAI API key)
load_dotenv()
# embeddings = OpenAIEmbeddings(api_key=os.getenv("openai_api_key"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Most powerful embeddings
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Create and persist Chroma vectorstore
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")
# vectorstore.persist()
# commented because data is auto-persisted on creation in latest versions

print("Vectorstore created and persisted in ./chroma_db")

