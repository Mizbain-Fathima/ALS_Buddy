from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# --- Embeddings + Vector store ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # keep context tight

t5_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the following question based on the context provided.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer in one short and clear sentence:"
    ),
)

# --- Flan-T5 (instruction-tuned) ---
model_name = "google/flan-t5-base"  # use flan-t5-small if RAM-limited
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,             # ensure encoder input <= 512 tokens
    max_new_tokens=128,          # decoder budget (output length)
    # do_sample=True,            # uncomment to enable sampling
    # temperature=0.3, top_p=0.95,
    device=0 if torch.cuda.is_available() else -1,
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Shorter memory window ---
memory = ConversationBufferWindowMemory(
    k=2,                         # small history to avoid overstuffing
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

# --- RAG chain with token cap on stuffed docs ---
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True,
    combine_docs_chain_kwargs={"prompt": t5_prompt},
)

# --- Test query ---
query = "What are the first symptoms of ALS?"
response = rag_chain.invoke({"question": query})
print("Query:", query)
print("Response:", response["answer"])
if "source_documents" in response:
    print("Sources:", [doc.page_content[:150] + "..." for doc in response["source_documents"]])
