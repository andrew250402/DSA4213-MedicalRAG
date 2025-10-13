"""
rag_pipeline.py
- Loads FAISS vectorstore (if RAG enabled)
- Connects retriever to Hugging Face local LLM
- Builds QA or direct LLM chain depending on ENABLE_RAG
"""

import config
import pickle
from langchain.chains import RetrievalQA, LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_vector_db():
    with open("vectorstore.pkl", "rb") as f:
        return pickle.load(f)


def build_llm():
    """Load Hugging Face model and wrap it with LangChain pipeline."""
    model_name = "google/flan-t5-base"  # ✅ CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )

    return HuggingFacePipeline(pipeline=pipe)


def build_rag_pipeline():
    """Builds a Retrieval-Augmented Generation (RAG) pipeline or fallback LLM."""
    llm = build_llm()

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

    # 🔀 Conditional RAG logic
    if getattr(config, "ENABLE_RAG", True):
        print("[INFO] RAG mode enabled — loading vector database...")
        db = load_vector_db()
        retriever = db.as_retriever(search_kwargs={"k": config.K})

        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT},
        )
    else:
        print("[INFO] RAG mode disabled — using LLM only.")
        DIRECT_PROMPT = PromptTemplate(
            template=(
                "You are a helpful medical assistant. Answer the question clearly and completely.\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
            input_variables=["question"],
        )
        return LLMChain(llm=llm, prompt=DIRECT_PROMPT)


if __name__ == "__main__":
    qa = build_rag_pipeline()

    question = "What are common symptoms of diabetes?"
    print(f"\n[QUESTION] {question}")

    if getattr(config, "ENABLE_RAG", True):
        response = qa.invoke(question)
    else:
        response = qa.invoke({"question": question})

    print("\n[ANSWER]", response)
