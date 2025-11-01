"""
build_bm25_store.py
- Creates or updates the BM25 store from text files in the data directory.
"""

import argparse
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bm25_vectorstore import BM25VectorStore

BM25_PATH = "store_v2/bm25"
DATA_PATH = "data_v2/plain"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing BM25 Database")
        clear_database()

    # Create (or update) the BM25 store.
    documents = load_documents()
    chunks = split_documents(documents)
    create_or_update_bm25_index(chunks)


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


def create_or_update_bm25_index(chunks: list[Document]):
    if os.path.exists(BM25_PATH):
        # Load the existing database
        print("Loading existing BM25 index...")
        db = BM25VectorStore.load_local(BM25_PATH, allow_dangerous_deserialization=True)

        # Calculate chunk IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Get existing IDs from the BM25 store
        existing_ids = set(db.get_all_document_ids())

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            # Add new documents to the existing index
            db.add_documents(new_chunks, ids=[
                             chunk.metadata["id"] for chunk in new_chunks])
            # Save the updated index
            db.save_local(BM25_PATH)
        else:
            print("✅ No new documents to add")

    else:
        # Create a new database from the documents
        print("Creating new BM25 index...")
        chunks_with_ids = calculate_chunk_ids(chunks)
        db = BM25VectorStore.from_documents(chunks_with_ids, ids=[
                                          chunk.metadata["id"] for chunk in chunks_with_ids])
        # Save the new index
        db.save_local(BM25_PATH)
        print(
            f"✅ Created and saved a new BM25 index with {len(chunks_with_ids)} documents.")


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
    if os.path.exists(BM25_PATH):
        shutil.rmtree(BM25_PATH)


if __name__ == "__main__":
    main()