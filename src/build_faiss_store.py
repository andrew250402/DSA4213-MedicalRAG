"""
build_faiss_store.py
- Creates or updates the vector store from text files in the data directory.
"""

import argparse
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data_v2/plain"

# FAISS_PATH = "store_v2/faiss/Bio_ClinicalBERT"
# embedding_model = HuggingFaceEmbeddings(
#     model_name="emilyalsentzer/Bio_ClinicalBERT"
# )

# FAISS_PATH = "store_v2/faiss/SapBERT-from-PubMedBERT-fulltext"
# embedding_model = HuggingFaceEmbeddings(
#     model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# )

# FAISS_PATH = "store_v2/faiss/MedEmbed-base-v0.1"
# embedding_model = HuggingFaceEmbeddings(
#     model_name="abhinand/MedEmbed-base-v0.1"
# )

# FAISS_PATH = "store_v2/faiss/pubmedbert-base-embeddings"
# embedding_model = HuggingFaceEmbeddings(
#     model_name="NeuML/pubmedbert-base-embeddings"
# )

FAISS_PATH = "store_v2/faiss/openai-text-embedding-3-small"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    create_or_update_faiss_index(chunks)


def load_documents():
    document_loader = DirectoryLoader(
        DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    return text_splitter.split_documents(documents)


def create_or_update_faiss_index(chunks: list[Document]):
    if os.path.exists(FAISS_PATH):
        # Load the existing database
        db = FAISS.load_local(FAISS_PATH, embedding_model,
                              allow_dangerous_deserialization=True)

        # Calculate chunk IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Get existing IDs from the FAISS index's docstore
        all_doc_ids = list(db.index_to_docstore_id.values())

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in all_doc_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            # Add new documents to the existing index
            db.add_documents(new_chunks, ids=[
                             chunk.metadata["id"] for chunk in new_chunks])
            # Save the updated index
            db.save_local(FAISS_PATH)
        else:
            print("✅ No new documents to add")

    else:
        # Create a new database from the documents
        print("Creating new FAISS index...")
        chunks_with_ids = calculate_chunk_ids(chunks)
        db = FAISS.from_documents(chunks_with_ids, embedding_model, ids=[
                                  chunk.metadata["id"] for chunk in chunks_with_ids])
        # Save the new index
        db.save_local(FAISS_PATH)
        print(
            f"✅ Created and saved a new index with {len(chunks_with_ids)} documents.")


def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each chunk based on its source.
    The ID format is: "source_file:chunk_index"
    """
    last_source = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        # We will use the source file path as the basis for the ID.
        current_source_id = f"{source}"

        # If the source is the same as the last one, increment the index.
        if current_source_id == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source = current_source_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)


if __name__ == "__main__":
    main()
