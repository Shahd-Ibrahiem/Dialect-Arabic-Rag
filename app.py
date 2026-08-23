import streamlit as st
from src.agent import DialectRAGAgent

st.set_page_config(page_title="Dialect-Aware Arabic RAG Agent", page_icon="🌙", layout="wide")

st.title("🌙 Dialect-Aware Arabic RAG Agent")
st.caption("Vector retrieval and grounding engine optimized for Arabic regional dialects.")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password")
    dialect = st.selectbox("Response Dialect", ["Egyptian (المصرية)", "Gulf (الخليجية)", "Levantine (الشمية)", "MSA (الفصحى)"])

# Document Ingestion Section
st.subheader("📚 Knowledge Base Ingestion")
doc_text = st.text_area("Add reference text/knowledge base in Arabic:", height=120, value="تكون ساعات العمل الرسمية في الشركة من الساعة 9 صباحاً حتى 5 مساءً. يتم تقديم طلبات الإجازات قبل اسبوع من موعد الإجازة عبر البوابة الإلكترونية.")

if st.button("📥 Index Document into Vector Database"):
    if doc_text:
        agent = DialectRAGAgent(api_key=groq_key)
        agent.vector_store.add_documents([doc_text])
        st.success("Document successfully indexed into ChromaDB vector store!")

st.divider()

# Query Section
st.subheader("💬 Query Knowledge Base")
query = st.text_input("Ask a question in your dialect:", value="ازاي اقدم على إجازة وفين؟")

if st.button("🚀 Ask Agent", type="primary"):
    if query and groq_key:
        with st.spinner("Searching vectors and generating dialect response..."):
            try:
                agent = DialectRAGAgent(api_key=groq_key)
                selected_dialect = dialect.split()[0]
                result = agent.answer_query(query=query, dialect=selected_dialect)

                st.markdown("### 💬 Answer:")
                st.write(result["answer"])

                with st.expander("🔍 Retrieved Vector Context"):
                    for idx, ctx in enumerate(result["retrieved_context"]):
                        st.info(f"**Match {idx+1}:** {ctx['text']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")