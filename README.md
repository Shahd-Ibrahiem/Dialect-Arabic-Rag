# 🌙 Dialect-Aware Arabic RAG Agent

An Intelligent Retrieval-Augmented Generation (RAG) system optimized for regional Arabic dialects (Egyptian, Gulf, Levantine, MSA). It addresses standard vector retrieval bottlenecks on dialectal text using specialized text normalization, query expansion, ChromaDB embeddings, and Groq LLMs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.0-orange.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-API-purple.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Standard multilingual embedding models and semantic search engines often underperform on regional Arabic dialects due to heavy Modern Standard Arabic (MSA) bias, diacritics, and varied spelling conventions. 

**Dialect-Aware Arabic RAG Agent** solves this challenge by introducing an end-to-end processing pipeline:
1. **Preprocessing Layer:** Normalizes orthographic variations (Alef, Yeh, Teh Marbuta) and strips Tashkeel.
2. **Query Expansion Engine:** Translates informal dialectal query terms into multi-token search terms to maximize vector recall.
3. **Multilingual Vector Retrieval:** Uses `paraphrase-multilingual-MiniLM-L12-v2` inside ChromaDB for semantic similarity search.
4. **Dialect-Grounded LLM Generation:** Formulates natural, dialect-specific responses strictly grounded in retrieved document context.

---

## ✨ Key Features

* 🗣️ **Regional Dialect Response Engine:** Supports targeted responses in Egyptian, Gulf, Levantine, and MSA registers.
* 🔍 **Dialectal Query Expansion:** Maps colloquial search terms (e.g., "ازاي", "ليه", "فين") to semantic search tokens prior to vector lookup.
* 🧹 **Arabic Text Normalization:** Cleans diacritics, character variations, and noise for uniform embedding indexation.
* ⚡ **ChromaDB Vector Store:** Local persistent vector database setup for low-latency similarity queries.
* 🚀 **Groq-Powered Generation:** Leverages fast Llama 3 models on Groq for grounded, context-aware answer synthesis.
* 📊 **Transparent Audit Trail:** Displays exact retrieved vector chunks alongside generated answers in the Streamlit UI.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
* **LLM Orchestration:** Groq API (`llama-3.3-70b-versatile`)
* **UI & Dashboard:** Streamlit
* **Environment:** `python-dotenv`

---

## 📂 Repository Structure

```text
dialect-arabic-rag/
├── .streamlit/
│   └── config.toml          # Custom Emerald theme styling for Streamlit
├── chroma_db/               # Persistent ChromaDB vector storage (git-ignored)
├── src/
│   ├── __init__.py          # Marks src as a Python package
│   ├── preprocessor.py      # Arabic text normalizer & dialect query expander
│   ├── vectorstore.py       # ChromaDB indexation & similarity search manager
│   └── agent.py             # Dialect RAG agent and Groq synthesis pipeline
├── .env                     # API key configuration (git-ignored)
├── .gitignore               # Ignored directories (venv, chroma_db, env)
├── app.py                   # Streamlit interactive dashboard
├── README.md                # Project documentation
└── requirements.txt         # Dependencies list