import os
import sys
from typing import List

from langchain.chat_models import init_chat_model  # OpenAI chat LLM
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import FAISS  # optional
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from bm25_vectorstore import BM25VectorStore
from dotenv import load_dotenv
load_dotenv()

import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======== VECTOR STORE ========

def load_vector_db():
    """Load FAISS or BM25 database based on config.RETRIEVER_TYPE"""
    retriever_type = getattr(config, "RETRIEVER_TYPE", "faiss").lower()

    if retriever_type == "faiss":
        print(f"[INFO] Loading FAISS DB from {config.FAISS_PATH}")
        embedding_model = init_embeddings(model="openai:text-embedding-3-small", api_key = os.environ.get("OPENAI_API_KEY"))
        # embedding_model = OpenAIEmbeddings("text-embedding-3-small")
        return FAISS.load_local(config.FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    elif retriever_type == "bm25":
        print(f"[INFO] Loading BM25 DB from {config.BM25_PATH}")
        return BM25VectorStore.load_local(config.BM25_PATH, allow_dangerous_deserialization=True)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# ======== LLM ========

def build_llm():
    """Load OpenAI chat LLM"""
    return init_chat_model(model="gpt-3.5-turbo", temperature=0)


# ======== RAG AGENT ========

def build_rag_agent(vector_store):
    """Builds a Retrieval-Augmented Generation agent"""
    llm = build_llm()

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant docs from vector store"""
        retrieved_docs = vector_store.similarity_search(query, k=getattr(config, "K", 2))
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]

    prompt = (
        "You have access to a tool that retrieves context from the knowledge base. "
        "Use it to answer user queries."
    )

    agent = create_agent(llm, tools=tools, system_prompt=prompt)
    return agent


# ======== EXAMPLE USAGE ========

if __name__ == "__main__":
    vector_store = load_vector_db()
    agent = build_rag_agent(vector_store)

    query = "What are common symptoms of diabetes?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
