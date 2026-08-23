import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from src.preprocessor import ArabicTextPreprocessor

class ArabicVectorStore:
    def __init__(self, collection_name: str = "arabic_dialect_docs"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Multi-lingual embedding model optimized for Arabic semantics
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Preprocesses and indexes documents into ChromaDB vectorstore."""
        processed_docs = [ArabicTextPreprocessor.normalize_arabic(d) for d in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        if not metadatas:
            metadatas = [{"source": "manual_entry"} for _ in documents]

        self.collection.add(
            documents=processed_docs,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes similarity search with query expansion."""
        expanded_query = ArabicTextPreprocessor.expand_dialect_query(query)
        normalized_query = ArabicTextPreprocessor.normalize_arabic(expanded_query)

        results = self.collection.query(
            query_texts=[normalized_query],
            n_results=top_k
        )
        
        matches = []
        if results and results["documents"] and len(results["documents"][0]) > 0:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                matches.append({"text": doc, "metadata": meta})
        return matches