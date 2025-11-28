from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.combine_documents import StuffDocumentsChain
from langchain_core.documents import Document
from .prompts import rag_prompt, support_prompt
from .rag_setup import get_retriever
from transformers import pipeline
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate

retriever = get_retriever()

hf_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=200,
    repetition_penalty=1.3,
    do_sample=False
)
hf_pipeline.tokenizer.pad_token_id = hf_pipeline.model.config.eos_token_id
hf_pipeline.model.config.use_cache = True
llm = HuggingFacePipeline(pipeline=hf_pipeline)

parser = StrOutputParser()

doc_chain = create_stuff_documents_chain(llm, rag_prompt)
# doc_chain = StuffDocumentsChain(
#     llm_chain=llm,
#     document_variable_name="context",
#     prompt=rag_prompt
# )

# ---- SUMMARY CHAIN ----
summary_chain = rag_prompt | llm | parser

# ---- SUPPORT CHAIN ----
support_chain = support_prompt | llm | parser

# ---- PARALLEL EXECUTION ----
parallel_chain = RunnableParallel(
    medical=summary_chain,
    support=support_chain
)

def answer_question(question: str) -> str:

    raw = retriever.invoke(question)

    # Guard if no docs
    if not raw:
        return "I don't know based on available data."

    # Normalize results into Documents
    docs = []
    for d in raw:
        if isinstance(d, Document):
            docs.append(d)
        else:
            docs.append(Document(page_content=str(d)))

    # Build context for parallel chains
    context = "\n\n".join(d.page_content for d in docs)

    # Run both chains in parallel
    result = parallel_chain.invoke({
        "context": context,
        "question": question
    })

    medical = result.get("medical", "")
    support = result.get("support", "").strip()

    # remove template echoes
    for bad in ["You are", "RULES:", "User question:", "Answer:"]:
        support = support.replace(bad, "").strip()

    # enforce line limit (max 5 lines)
    support_lines = support.splitlines()
    support = "\n".join(support_lines[:5])


    # Clean TinyLlama tokens
    if "<|assistant|>" in medical:
        medical = medical.split("<|assistant|>")[-1].strip()
    if "<|assistant|>" in support:
        support = support.split("<|assistant|>")[-1].strip()

    return f"""
Medical Summary:
{medical[:800]}

Support Response:
{support}
"""

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(answer_question(q))
