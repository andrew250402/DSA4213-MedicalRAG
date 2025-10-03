"""
chatbot_cli.py
- CLI interface for RAG chatbot
"""

from rag_pipeline import build_rag_pipeline

def run_chatbot():
    qa = build_rag_pipeline()
    print("Medical RAG Chatbot (type 'exit' to quit)")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response = qa.invoke(query)
        print("Bot:", response)

if __name__ == "__main__":
    run_chatbot()
