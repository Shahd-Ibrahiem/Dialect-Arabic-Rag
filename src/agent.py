import os
from groq import Groq
from src.vectorstore import ArabicVectorStore
from src.preprocessor import ArabicTextPreprocessor

class DialectRAGAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.vector_store = ArabicVectorStore()

    def _get_active_model(self) -> str:
        """Dynamically retrieves the first available text chat model from Groq."""
        try:
            available_models = self.client.models.list()
            # Filter for text models, excluding audio/whisper models
            chat_models = [
                m.id for m in available_models.data 
                if not any(sub in m.id for sub in ["whisper", "guard", "vision"])
            ]
            
            # Prefer standard production models if available
            preferred = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "qwen-2.5-32b"]
            for pref in preferred:
                if pref in chat_models:
                    return pref
            
            if chat_models:
                return chat_models[0]
        except Exception:
            pass
        
        # Default fallback string
        return "llama3-8b-8192"

    def answer_query(self, query: str, dialect: str = "Egyptian") -> dict:
        if not self.client:
            raise ValueError("Groq API Key is missing. Please provide a valid key.")

        # 1. Retrieve context
        retrieved_docs = self.vector_store.search(query, top_k=3)
        context_str = "\n".join([d["text"] for d in retrieved_docs]) if retrieved_docs else "لا يوجد سياق مباشر."

        # 2. System prompt
        system_prompt = (
            f"You are an expert AI assistant fluent in regional Arabic dialects. "
            f"Answer the user's question accurately using only the provided context. "
            f"Respond exclusively in the {dialect} dialect."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        # 3. Dynamic Model Selection & API Call
        model_id = self._get_active_model()

        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "retrieved_context": retrieved_docs,
            "model_used": model_id
        }