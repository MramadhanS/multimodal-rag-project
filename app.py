import os
import sys
import time
import streamlit as st
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from agent.brain import ResearcherAgentOllama
from utils.mlops_logger import log_agent_interaction

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

st.set_page_config(page_title="InsightMind // RAG", page_icon="⚡", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #212121 !important; color: #ececec !important; }
    [data-testid="stSidebar"] { background-color: #171717 !important; border-right: 1px solid #2f2f2f; }
    .stChatInput { border-radius: 24px !important; background-color: #2f2f2f !important; }
    h1, h2, h3, p { color: #ececec !important; }
    .stExpander { background-color: #2f2f2f !important; border: 1px solid #424242 !important; border-radius: 8px !important; }
    div[data-testid="stChatMessage"] { border-radius: 12px !important; padding: 15px !important; margin-bottom: 10px !important; }
    div[data-testid="stChatMessageUser"] { background-color: #2f2f2f !important; }
    div[data-testid="stChatMessageAssistant"] { background-color: transparent !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ arXiv CS Research Assistant")
st.caption("Chat dan Eksplor Paper Terbaru AI, NLP, dan Computer Vision")

# State Management Memori Chat
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = ResearcherAgentOllama()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SIDEBAR MINIMALIS
with st.sidebar:
    st.markdown("<h3 style='font-weight:700;'>📁 Kontrol Panel</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload Gambar Diagram, Bar Chart, Metrik dan lain lain:", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Gambar Siap Dikirim", use_column_width=True)
    st.markdown("---")
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# RENDER CHAT LAMA
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "img_path" in msg and msg["img_path"]:
            if os.path.exists(msg["img_path"]):
                st.image(msg["img_path"], width=450)
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🛠️ Log Perangkat", expanded=False):
                st.json(msg["trace"])

# LOGIKA INPUT CHAT BARU
user_input = st.chat_input("Ask Here")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = ""
        final_trace = []
        detected_file_path = None
        detected_caption = ""
        is_visual_requested = False

        temp_image_path = None
        if uploaded_image:
            temp_image_path = os.path.join(BASE_DIR, "data", f"temp_{int(time.time())}.png")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

        start_time = time.time()
        
        for chunk_data in st.session_state.agent_instance.ask(user_input, temp_image_path):
            if chunk_data["status"] == "processing":
                full_response_text += chunk_data["text_chunk"]
                clean_display_text = full_response_text.replace("```markdown", "").replace("```", "")
                message_placeholder.markdown(clean_display_text + "⏳")
                final_trace = chunk_data["trace"]
                is_visual_requested = chunk_data["needs_visual"]
                
                if "active_image_paths" in chunk_data and chunk_data["active_image_paths"]:
                    # Mengambil gambar pertama yang dibaca oleh AI
                    potential_path = chunk_data["active_image_paths"][0]
                    if os.path.exists(potential_path):
                        detected_file_path = potential_path

                if chunk_data["reverse_search_data"] and chunk_data["reverse_search_data"]["success"]:
                    detected_file_path = chunk_data["reverse_search_data"]["file_path"]
                    detected_caption = chunk_data["reverse_search_data"]["caption"]

        # Setelah streaming selesai, matikan loading
        message_placeholder.markdown(full_response_text.replace("```markdown", "").replace("```", ""))
        latency = time.time() - start_time

        if detected_file_path and os.path.exists(detected_file_path):
            st.markdown("---")
            st.image(
                detected_file_path, 
                caption="🖼️ Bukti Visual Dokumen Jurnal yang Dianalisis oleh AI", 
                width=550
            )
        elif is_visual_requested:
            st.warning("ℹ *Sistem tidak menemukan berkas diagram visual yang cocok di database paper.*")

        final_clean_text = full_response_text.replace("```markdown", "").replace("```", "")
        message_placeholder.markdown(final_clean_text)
        latency = time.time() - start_time

        if is_visual_requested and not detected_file_path:
            try:
                # 1. Optimasi kata kunci pencarian agar dipahami oleh model CLIP OpenAI
                clip_query = user_input
                if "arsitektur" in clip_query.lower() or "architecture" in clip_query.lower():
                    clip_query += " architecture framework pipeline layout"
                if "gambar" in clip_query.lower() or "diagram" in clip_query.lower():
                    clip_query += " diagram flowchart schematic plot"

                # 2. Panggil fungsi pencari gambar dari Qdrant
                img_check_result = st.session_state.agent_instance.db.search_images_via_clip(clip_query, limit=1)
                
                # 3. BARIS PERBAIKAN TOTAL: Parsing string multi-baris secara aman
                if img_check_result and "IMAGE_PATH:" in img_check_result:
                    for line in img_check_result.split("\n"):
                        clean_line = line.strip()
                        
                        # Deteksi baris yang berisi alamat path fisik file gambar
                        if "IMAGE_PATH:" in clean_line:
                            parts = clean_line.split("IMAGE_PATH:", 1)
                            if len(parts) > 1:
                                potential_path = parts[1].strip()
                                if os.path.exists(potential_path):
                                    detected_file_path = potential_path
                        
                        if "CAPTION:" in clean_line:
                            parts = clean_line.split("CAPTION:", 1)
                            if len(parts) > 1:
                                detected_caption = parts[1].strip()

            except Exception as e:
                print(f"❌ Gagal memproses pencarian gambar internal: {e}")

        if detected_file_path and os.path.exists(detected_file_path):
            st.markdown("---")
            st.image(
                detected_file_path, 
                caption=f"🖼️ Aset Visual Terkait: {detected_caption}", 
                width=500
            )
        elif is_visual_requested:
            st.warning("ℹ️ *Sistem tidak menemukan berkas diagram/gambar yang relevan di dalam database paper Anda.*")


        # TAMPILKAN LOG JALURNYA
        if final_trace:
            with st.expander("🛠️ Log Perangkat", expanded=False):
                st.json(final_trace)
                
        st.caption(f"⏱️ Lama Durasi Respons: {latency:.2f} detik")

    # KUNCI RIWAYAT DATA KE UI
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": full_response_text,
        "trace": final_trace,
        "img_path": detected_file_path if is_visual_requested or uploaded_image else None
    })

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)
        
    log_agent_interaction(user_input, full_response_text, final_trace, latency)
