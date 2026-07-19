import os
import sys
import json
import logging
import time
import ollama
import base64
from typing import Dict, Any, Generator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from agent.qdrant_client_tools import MultimodalQdrantRetriever

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/ollama_agent.log"),
    level=logging.INFO,
    format="%(asctime)s | [OLLAMA_VISION_ACTIVE] | %(message)s"
)
logger = logging.getLogger(__name__)

class ResearcherAgentOllama:
    def __init__(self):
        self.model_name = "qwen2.5vl:7b"
        self.db = MultimodalQdrantRetriever()
        
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = ollama_host
        
        import ollama
        
        self.system_prompt = """
Anda adalah AI Peneliti Literatur Ilmiah tingkat doktoral yang memiliki kemampuan analisis visual dokumen (Vision) yang luar biasa tajam. 
Tugas Anda adalah membedah dan merangkum isi data teks, tabel metrik, maupun diagram gambar yang disediakan di bawah ini secara objektif dan profesional.

SOP PENALARAN VISUAL & TEKS:
1. PANDANGI GAMBAR: Jika ada gambar/diagram yang terlampir secara visual, amati pikselnya dengan seksama lalu jabarkan apa maksud dari alur diagram atau grafik tersebut.
2. WAJIB SITASI JURNAL: Sertakan ID Paper (Contoh: [PAPER ID: 2607.13558v1]) di setiap kesimpulan atau rangkuman Anda.
3. ANTI-HALUSINASI: Jika informasi atau gambar yang diminta benar-benar tidak ada di data pendukung, katakan dengan jujur data tidak tersedia.
4. JIKA DIMINTA TABEL: Anda WAJIB menyajikan data evaluasi tersebut menggunakan format tabel Markdown standar murni dengan pembatas pipa (contoh: | Kolom 1 | Kolom 2 |). JANGAN PERNAH menambahkan teks pembungkus kode seperti ```markdown atau spasi tab liar di awal baris tabel karena akan merusak sistem visual layar pengguna!
"""

    def ask(self, user_query: str, uploaded_image_path: str = None) -> Generator[Dict[str, Any], None, None]:
        logger.info(f"Kueri Qwen-Vision: '{user_query}'")
        self.current_trace = []
        context_text = ""
        context_visual_info = ""
        image_attachments = []
        ollama_images_base64 = []
        reverse_result = None
        needs_visual = False

        try:
            # JALUR B: REVERSE IMAGE SEARCH
            if uploaded_image_path:
                print("📸 USER UPLOAD GAMBAR: Memicu mesin pencari gambar kembar Qdrant...")
                reverse_result = self.db.search_by_uploaded_image(uploaded_image_path, limit=1)
                if reverse_result["success"]:
                    context_visual_info = reverse_result["message"]
                    image_attachments.append(reverse_result["file_path"])
                    self.current_trace.append({
                        "step": "REVERSE_IMAGE_SEARCH",
                        "tool_called": "search_by_uploaded_image",
                        "arguments": {"image_path": uploaded_image_path}
                    })
                else:
                    context_visual_info = reverse_result["message"]
                context_text = self.db.search_text_and_tables(query=user_query, limit=2)
                
            # JALUR A: TEXT INTENT SELECTION
            else:
                visual_keywords = ["show", "tunjukkan", "gambar", "diagram", "chart", "flowchart", "figure", "fig", "illustration", "plot", "grafik"]
                needs_visual = any(kw in user_query.lower() for kw in visual_keywords)
                
                context_text = self.db.search_text_and_tables(query=user_query, limit=2)
                self.current_trace.append({
                    "step": "ACTION",
                    "tool_called": "search_text_and_tables",
                    "arguments": {"query": user_query, "limit": 2}
                })

                if needs_visual:
                    context_visual_info = self.db.search_images_via_clip(query=user_query, limit=1)
                    for line in context_visual_info.split("\n"):
                        if "IMAGE_PATH:" in line:
                            path_extracted = line.replace("IMAGE_PATH:", "").strip()
                            if os.path.exists(path_extracted):
                                image_attachments.append(path_extracted)
                    self.current_trace.append({
                        "step": "ACTION",
                        "tool_called": "search_images_via_clip",
                        "arguments": {"query": user_query, "limit": 1}
                    })

            # Konversi gambar fisik ke string Base64 murni
            if uploaded_image_path and (uploaded_image_path not in image_attachments):
                image_attachments.append(uploaded_image_path)

            for img_path in image_attachments:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        ollama_images_base64.append(encoded_string)

            # Merakit prompt komparasi riset
            full_prompt = (
                f"=== DATA TEKS & TABEL QDRANT ===\n{context_text}\n\n"
                f"=== METADATA VISUAL QDRANT ===\n{context_visual_info}\n\n"
                f"PERTANYAAN USER: {user_query}"
            )
            
            print("🧠 Membuka mata Qwen-VL lokal via Pipa Chat Streaming...")
            
            messages_payload = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt, "images": ollama_images_base64 if ollama_images_base64 else None}
            ]
            
            stream = ollama.chat(
                model=self.model_name,
                messages=messages_payload,
                stream=True,
                options={"num_ctx": 16384}
            )
            
            for chunk in stream:
                # Menyadap potongan teks dari format kembalian chat resmi Ollama
                if 'message' in chunk and 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    yield {
                        "status": "processing",
                        "text_chunk": text_chunk,
                        "trace": self.current_trace,
                        "needs_visual": needs_visual,
                        "reverse_search_data": reverse_result if uploaded_image_path else None
                    }
                
        except Exception as e:
            logger.error(f"Error pada Agent Vision: {e}")
            yield {"status": "error", "text_chunk": f"Terjadi kesalahan fatal: {str(e)}", "trace": []}

if __name__ == "__main__":
    agent = ResearcherAgentOllama()
    print("Menjalankan Simulasi Uji Coba Chat Streaming LLaVA...")
    for chunk_data in agent.ask("Apa definisi dari computer vision"):
        sys.stdout.write(chunk_data["text_chunk"])
        sys.stdout.flush()
    print("\n\nTest Selesai.")
