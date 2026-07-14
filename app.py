# Import Library
import os 
import sys  
import time  
import streamlit as st  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from agent.brain import ResearcherAgentOllama
from utils.mlops_logger import log_agent_interaction

# Streamlit UI Configuration
st.set_page_config(
    page_title="Multimodal Agentic RAG",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Academic Research Assistant (Agentic RAG)")
st.caption("Powered by Ollama Llama3, Docker Qdrant Vector Search, and ReAct Intent Framework.")

# State Management
if "agent_instance" not in st.session_state:

    with st.spinner("Menginisialisasi sistem saraf agen AI lokal..."):
        try:
            st.session_state.agent_instance = ResearcherAgentOllama()
        
        except Exception as e:
            st.error(f"Sistem gagal menyala. Pastikan Docker Qdrant dan Ollama Anda aktif! Detail: {e}")
            st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Rendering History Message
for msg in st.session_state.chat_history :

    with st.chat_message(msg["role"]) :
        st.markdown(msg["content"])

    if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("Tampilkan Reasoning Trace (Penggunaan Tool)", expanded=False):
                for step in msg["trace"]:
                    if step.get("step") == "SKIP_ACTION":
                        st.code(f"Alat Di-Skip: {step['tool_called']}\nAlasan: {step['reason']}", language="text")
                    else:
                        st.code(f"Alat Digunakan: {step['tool_called']}\nParameter: {step['arguments']}", language="json")

# User Input
user_input = st.chat_input ("Tanyakan Sesuatu Tentang Literatur Penelitian")

if user_input :
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agen sedang menyisir laci Qdrant dan menyintesis jawaban...") :
        
            agent = st.session_state.agent_instance
            start_time = time.time()
            answer, trace = agent.ask(user_input)
            latency = time.time() - start_time
            
            st.markdown(answer)

            if trace:
                with st.expander("Tampilkan Reasoning Trace (Penggunaan Tool)", expanded=True):
                    for step in trace:
                        if step.get("step") == "SKIP_ACTION":
                            st.code(f"Alat Di-Skip: {step['tool_called']}\nAlasan: {step['reason']}", language="text")
                        else:
                            st.code(f"Alat Digunakan: {step['tool_called']}\nParameter: {step['arguments']}", language="json")
            
            st.caption(f"⏱️ Waktu sintesis: {latency:.2f} detik")

    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": answer,
        "trace": trace
    })
    
    log_agent_interaction(user_input, answer, trace, latency)