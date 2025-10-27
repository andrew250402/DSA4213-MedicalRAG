import os
import sys
import logging
from datetime import datetime
from typing import List

from langchain.chat_models import init_chat_model  # OpenAI chat LLM
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
from prompt_templates import build_prompt
from bm25_vectorstore import BM25VectorStore

import config

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======== SETUP LOGGING ========

def setup_logger():
    """Configure logging for RAG runs."""
    logger = logging.getLogger("RAGLogger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if re-run
    if not logger.handlers:
        handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# ======== VECTOR STORE ========

def load_vector_db():
    retriever_type = getattr(config, "RETRIEVER_TYPE", "faiss").lower()

    if retriever_type == "faiss":
        print(f"[INFO] Loading FAISS DB from {config.FAISS_PATH}")
        embedding_model = init_embeddings(
            model="openai:text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        return FAISS.load_local(config.FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    elif retriever_type == "bm25":
        print(f"[INFO] Loading BM25 DB from {config.BM25_PATH}")
        return BM25VectorStore.load_local(config.BM25_PATH, allow_dangerous_deserialization=True)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# ======== LLM ========

def build_llm():
    return init_chat_model(model=config.MODEL, temperature=0)


# ======== RAG AGENT ========

def build_rag_agent(vector_store):
    llm = build_llm()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant docs from vector store"""
        retrieved_docs = vector_store.similarity_search(query, k=getattr(config, "K", 2))
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        prompt_text = build_prompt(config.PROMPT_STYLE, query, retrieved_docs)
        return prompt_text, retrieved_docs

    tools = [retrieve_context]

    system_prompt = (
        f"You are a medical RAG assistant using the '{config.PROMPT_STYLE}' prompt style. "
        "Use retrieved context to generate concise, evidence-grounded answers."
    )

    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)
    return agent


# ======== DIRECT LLM (no RAG) ========

def build_direct_agent():
    llm = build_llm()
    prompt = "You are a helpful assistant. Answer user queries to the best of your ability."
    agent = create_agent(llm, tools=[], system_prompt=prompt)
    return agent


# ======== UNIFIED AGENT BUILDER ========

def build_agent():
    if getattr(config, "ENABLE_RAG", True):
        print("[INFO] RAG is ENABLED - Loading vector store...")
        vector_store = load_vector_db()
        return build_rag_agent(vector_store)
    else:
        print("[INFO] RAG is DISABLED - Using direct LLM...")
        return build_direct_agent()


# ======== MAIN / LOGGING RUN ========

if __name__ == "__main__":
    agent = build_agent()

    query = config.QUERY
    logger.info("=== NEW QUERY SESSION START ===")
    logger.info(f"Model: {config.MODEL}")
    logger.info(f"Retriever Type: {config.RETRIEVER_TYPE}")
    logger.info(f"RAG Enabled: {config.ENABLE_RAG}")
    logger.info(f"K Documents: {config.K}")
    logger.info(f"Prompt Style: {config.PROMPT_STYLE}")
    logger.info(f"Query: {query}")

    # Stream and capture output
    full_answer = ""
    retrieved_docs_str = ""

    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]

        # Tool message (retrieval context)
        if hasattr(msg, "type") and msg.type == "tool":
            retrieved_docs = getattr(msg, "artifact", None)
            if retrieved_docs:
                retrieved_docs_str = "\n\n".join(
                    f"- {d.page_content[:300]}... (Source: {d.metadata})"
                    for d in retrieved_docs
                )
                logger.info(f"Retrieved Documents:\n{retrieved_docs_str}")

        # Assistant message (final LLM output)
        elif msg.__class__.__name__ == "AIMessage":
            full_answer += msg.content

    logger.info(f"Answer:\n{full_answer.strip()}")
    logger.info("=== END OF QUERY ===\n")
    print("Answer:", full_answer.strip())
