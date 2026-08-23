import os
from typing import Dict, Any, List
from groq import Groq
from src.vectorstore import ArabicVectorStore

class DialectRAGAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.vector_store = ArabicVectorStore()

    def answer_query(self, query: str, dialect: str = "Egyptian") -> Dict[str, Any]:
        if not self.client:
            raise ValueError("Groq API Key is missing.")

        # 1. Retrieve Context
        context_matches = self.vector_store.search(query, top_k=3)
        context_str = "\n---\n".join([m["text"] for m in context_matches]) if context_matches else "لا يتوفر سياق."

        # 2. Construct Dialect-Aware Prompt
        system_prompt = f"""
        أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة.
        يجب أن تكون إجابتك بالعامية ({dialect}) بشكل طبيعي ودقيق وبناءً على السياق فقط.

        السياق المتاح:
        {context_str}
        """

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )

        return {
            "answer": response.choices[0].message.content,
            "retrieved_context": context_matches
        }