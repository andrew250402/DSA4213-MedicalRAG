"""
chatbot_cli.py
CLI interface for Medical RAG using LangChain 1.x
"""

import sys
import os
import config
from rag_pipeline import build_rag_agent, build_direct_agent, load_vector_db

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Check if RAG is enabled
    if getattr(config, "ENABLE_RAG", True):
        print("[INFO] RAG ENABLED - Loading vector store...")
        vectorstore = load_vector_db()
        agent = build_rag_agent(vectorstore)
    else:
        print("[INFO] RAG DISABLED - Using direct LLM...")
        agent = build_direct_agent()
    
    print("Medical RAG CLI ready! Type your questions below (type 'exit' to quit).")
    print(f"Using model {config.MODEL} with prompt style {config.PROMPT_STYLE}")

    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break

        # One-shot invoke
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print("\n[ANSWER]")
        # response is a list of messages; get last message
        print(response["messages"][-1].text)


if __name__ == "__main__":
    main()