import math
from collections import Counter
from typing import List, Dict, Any
from src.preprocessor import ArabicTextPreprocessor

class ArabicVectorStore:
    def __init__(self, collection_name: str = "arabic_dialect_docs"):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def _compute_tf(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        total_terms = len(tokens)
        if total_terms == 0:
            return {}
        counts = Counter(tokens)
        return {term: count / total_terms for term, count in counts.items()}

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([val ** 2 for val in vec1.values()])
        sum2 = sum([val ** 2 for val in vec2.values()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        return numerator / denominator

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Preprocesses and indexes documents into local memory."""
        for idx, doc in enumerate(documents):
            processed_doc = ArabicTextPreprocessor.normalize_arabic(doc)
            self.documents.append(processed_doc)
            
            meta = metadatas[idx] if metadatas else {"source": "manual_entry"}
            self.metadatas.append(meta)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes similarity search with dialect query expansion."""
        if not self.documents:
            return []

        expanded_query = ArabicTextPreprocessor.expand_dialect_query(query)
        normalized_query = ArabicTextPreprocessor.normalize_arabic(expanded_query)
        
        query_tf = self._compute_tf(normalized_query)
        scores = []

        for idx, doc in enumerate(self.documents):
            doc_tf = self._compute_tf(doc)
            score = self._cosine_similarity(query_tf, doc_tf)
            scores.append((score, doc, self.metadatas[idx]))

        # Sort documents by similarity score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        matches = []
        for score, doc, meta in scores[:top_k]:
            matches.append({"text": doc, "metadata": meta})
            
        return matches