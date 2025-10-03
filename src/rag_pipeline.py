"""
rag_pipeline.py
- Loads FAISS vectorstore
- Connects retriever to Hugging Face local LLM
- Builds QA chain with custom prompt
"""

import pickle
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def load_vector_db():
    with open("vectorstore.pkl", "rb") as f:
        return pickle.load(f)


def build_rag_pipeline():
    # Load vector DB
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Load Hugging Face model locally
    model_name = "google/flan-t5-base"   # ✅ bigger than small, but still CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Custom QA prompt
    QA_PROMPT = PromptTemplate(
        template=(
            "You are a helpful medical assistant. "
            "Use the context provided to answer the question clearly and completely.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


if __name__ == "__main__":
    qa = build_rag_pipeline()
    response = qa.invoke("What are common symptoms of diabetes?")
    print(response)
