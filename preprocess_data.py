import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('webscrapped-data/als_articles_expanded.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

all_text = " ".join([d["content"] for d in data])

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(all_text)

print("Number of chunks:", len(chunks))
