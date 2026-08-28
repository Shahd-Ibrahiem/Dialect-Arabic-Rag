import streamlit as st
from src.agent import DialectRAGAgent

st.set_page_config(page_title="Dialect-Aware Arabic RAG Agent", page_icon="🌙", layout="wide")

st.title("🌙 Dialect-Aware Arabic RAG Agent")
st.caption("Vector retrieval and grounding engine optimized for Arabic regional dialects.")

# 1. Initialize Agent in Session State
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    dialect = st.selectbox("Response Dialect", ["Egyptian (المصرية)", "Gulf (الخليجية)", "Levantine (الشمية)", "MSA (الفصحى)"])

if "agent" not in st.session_state:
    try:
        st.session_state.agent = DialectRAGAgent(api_key=groq_key if groq_key else None)
    except Exception as e:
        st.error(f"Initialization Error: {e}")

# Update API key dynamically if entered
if groq_key and st.session_state.agent:
    st.session_state.agent.api_key = groq_key
    if st.session_state.agent.client is None:
        from groq import Groq
        st.session_state.agent.client = Groq(api_key=groq_key)

# Document Ingestion Section
st.subheader("📚 Knowledge Base Ingestion")
doc_text = st.text_area(
    "Add reference text/knowledge base in Arabic:", 
    height=120, 
    value="تكون ساعات العمل الرسمية في الشركة من الساعة 9 صباحاً حتى 5 مساءً. يتم تقديم طلبات الإجازات قبل اسبوع من موعد الإجازة عبر البوابة الإلكترونية."
)

if st.button("📥 Index Document into Vector Database", type="secondary"):
    if not doc_text.strip():
        st.warning("Please enter some text to index.")
    else:
        with st.spinner("Indexing text into ChromaDB..."):
            try:
                st.session_state.agent.vector_store.add_documents([doc_text])
                st.success("✅ Document successfully indexed into ChromaDB vector store!")
            except Exception as e:
                st.error(f"❌ Indexing Failed: {str(e)}")

st.divider()

# Query Section
st.subheader("💬 Query Knowledge Base")
query = st.text_input("Ask a question in your dialect:", value="ازاي اقدم على إجازة وفين؟")

if st.button("🚀 Ask Agent", type="primary"):
    if not groq_key:
        st.error("⚠️ Please enter your Groq API Key in the sidebar first!")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching vectors and generating dialect response..."):
            try:
                selected_dialect = dialect.split()[0]
                result = st.session_state.agent.answer_query(query=query, dialect=selected_dialect)

                st.markdown("### 💬 Answer:")
                st.write(result["answer"])

                with st.expander("🔍 Retrieved Vector Context"):
                    if result["retrieved_context"]:
                        for idx, ctx in enumerate(result["retrieved_context"]):
                            st.info(f"**Match {idx+1}:** {ctx['text']}")
                    else:
                        st.warning("No relevant context found in vector database.")
            except Exception as e:
                st.error(f"❌ Execution Error: {str(e)}")