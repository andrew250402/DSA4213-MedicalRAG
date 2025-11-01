"""
bm25_vectorstore.py
- BM25-based vectorstore implementation with interface similar to FAISS
- Provides sparse retrieval using BM25 algorithm
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import nltk
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class BM25VectorStore:
    """
    BM25-based vectorstore that provides sparse retrieval functionality.
    Implements a similar interface to FAISS for compatibility.
    """

    def __init__(self, documents: Optional[List[Document]] = None,
                 bm25_index: Optional[BM25Okapi] = None,
                 document_store: Optional[List[Document]] = None):
        """
        Initialize BM25VectorStore.

        Args:
            documents: List of documents to create index from
            bm25_index: Pre-built BM25 index
            document_store: List of documents corresponding to the index
        """
        self.bm25_index = bm25_index
        self.document_store = document_store or []

        if documents and not bm25_index:
            self._build_index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization and lowercasing
        tokens = nltk.word_tokenize(text.lower())
        # Remove punctuation and short tokens
        tokens = [token for token in tokens if token.isalnum()
                  and len(token) > 2]
        return tokens

    def _build_index(self, documents: List[Document]):
        """Build BM25 index from documents."""
        self.document_store = documents

        # Tokenize all documents
        tokenized_docs = []
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            tokenized_docs.append(tokens)

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)

    @classmethod
    def from_documents(cls, documents: List[Document],
                       embedding=None,  # Not used but kept for interface compatibility
                       ids: Optional[List[str]] = None) -> 'BM25VectorStore':
        """
        Create BM25VectorStore from documents.

        Args:
            documents: List of documents
            embedding: Not used (kept for compatibility)
            ids: Document IDs (stored in metadata)
        """
        # If IDs are provided, add them to metadata
        if ids:
            for doc, doc_id in zip(documents, ids):
                doc.metadata['id'] = doc_id

        return cls(documents=documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents using BM25.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of top-k most similar documents
        """
        if not self.bm25_index:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-k documents by score
        top_indices = scores.argsort()[::-1][:k]

        # Return corresponding documents
        return [self.document_store[i] for i in top_indices if i < len(self.document_store)]

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None):
        """
        Add new documents to the index.

        Args:
            documents: List of new documents
            ids: Optional list of document IDs
        """
        # Add IDs to metadata if provided
        if ids:
            for doc, doc_id in zip(documents, ids):
                doc.metadata['id'] = doc_id

        # Add documents to store
        self.document_store.extend(documents)

        # Rebuild index with all documents
        self._build_index(self.document_store)

    def save_local(self, path: str):
        """
        Save BM25 index and documents to local directory.

        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)

        # Save BM25 index
        with open(os.path.join(path, 'bm25_index.pkl'), 'wb') as f:
            pickle.dump(self.bm25_index, f)

        # Save documents
        documents_data = []
        for doc in self.document_store:
            documents_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })

        with open(os.path.join(path, 'documents.json'), 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_local(cls, path: str,
                   embedding=None,  # Not used but kept for interface compatibility
                   allow_dangerous_deserialization: bool = False) -> 'BM25VectorStore':
        """
        Load BM25 index and documents from local directory.

        Args:
            path: Directory path to load from
            embedding: Not used (kept for compatibility)
            allow_dangerous_deserialization: Required for pickle loading

        Returns:
            BM25VectorStore instance
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "You must set allow_dangerous_deserialization=True to load BM25 index")

        # Load BM25 index
        with open(os.path.join(path, 'bm25_index.pkl'), 'rb') as f:
            bm25_index = pickle.load(f)

        # Load documents
        with open(os.path.join(path, 'documents.json'), 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        # Recreate Document objects
        documents = []
        for doc_data in documents_data:
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            documents.append(doc)

        return cls(bm25_index=bm25_index, document_store=documents)

    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the store."""
        return [doc.metadata.get('id', '') for doc in self.document_store if doc.metadata.get('id')]
