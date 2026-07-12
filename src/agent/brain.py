import os
import json
import sys
import logging
import time
import ollama 
from typing import Tuple, List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from agent.qdrant_client_tools import MultimodalQdrantRetriever

# FUNGSI: Mengunci folder utama proyek dan menyiapkan sistem perekaman log teks.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/ollama_agent.log"),
    level=logging.INFO,
    format="%(asctime)s | [OLLAMA_RESEARCHER] | %(message)s"
)
logger = logging.getLogger(__name__)

class ResearcherAgentOllama:
    def __init__(self):
        self.model_name = "llama3"
        self.current_trace = [] 
        
        self.db = MultimodalQdrantRetriever()
        
        self.system_prompt = """
Anda adalah AI Peneliti Literatur Ilmiah tingkat doktoral yang sangat jujur dan akademis.
Tugas Anda adalah menjawab pertanyaan user berdasarkan data teks pendukung yang disediakan di bawah.

ATURAN SOP KETAT:
1. NO HALLUCINATION: JANGAN PERNAH mengarang data, angka metrik, atau spesifikasi paper di luar data yang disediakan.
2. WAJIB SITASI: Anda WAJIB menyertakan nomor identitas paper (contoh: [PAPER ID: 2606.23943v1]) di setiap klaim atau kutipan.
3. JIKA DATA TIDAK ADA: Jika data pendukung di bawah tidak berisi informasi yang diminta, katakan dengan jujur: "Data tersebut belum tersedia di database saya." Jangan mengarang jawaban palsu!
"""

    def ask(self, user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Pintu gerbang interaksi utama chatbot RAG berbasis Ollama lokal.
        Otomatis menyeleksi apakah butuh mencari gambar atau tidak.
        """
        logger.info(f"Menerima kueri lokal: '{user_query}'")
        self.current_trace = [] 
        
        try:
            start_time = time.time()
            
            # --- TAHAP 1: SELEKSI NIAT OTOMATIS (INTENT CLASSIFIER) ---
            visual_keywords = ["show", "tunjukkan", "gambar", "diagram", "chart", "flowchart", "figure", "fig", "illustration", "plot", "grafik"]
            needs_visual = any(kw in user_query.lower() for kw in visual_keywords)
            
            # --- TAHAP 2: BERBURU TEKS & TABEL (SELALU AKTIF) ---
            print("🔍 Berburu literatur teks dan tabel metrik di Qdrant...")
            
            # ▲ PERBAIKAN SINTAKSIS: Memanggil fungsi lewat variabel internal 'self.db' dengan parameter kueri yang jelas.
            context_text = self.db.search_text_and_tables(query=user_query, limit=3)
            
            self.current_trace.append({
                "step": "ACTION",
                "tool_called": "search_text_and_tables",
                "arguments": {"query": user_query, "limit": 3}
            })

            # --- TAHAP 3: BERBURU GAMBAR (HANYA AKTIF JIKA DIPERLUKAN) ---
            context_visual = "PENGUMUMAN: Pengguna tidak meminta lampiran visual/gambar untuk pertanyaan ini."
            
            if needs_visual:
                print("📸 PERTANYAAN DETEKSI VISUAL: Berburu gambar diagram via CLIP AI...")
                
                # ▲ PERBAIKAN SINTAKSIS: Memanggil fungsi visual lewat variabel internal 'self.db' secara mandiri.
                context_visual = self.db.search_images_via_clip(query=user_query, limit=2)
                
                self.current_trace.append({
                    "step": "ACTION",
                    "tool_called": "search_images_via_clip",
                    "arguments": {"query": user_query, "limit": 2}
                })
            else:
                self.current_trace.append({
                    "step": "SKIP_ACTION",
                    "tool_called": "search_images_via_clip",
                    "reason": "Kueri murni teks teoritis (Tidak mendeteksi kebutuhan visual)"
                })

            # --- TAHAP 4: MULTIMODAL CONTEXT INJECTION & GENERATION ---
            full_prompt = (
                f"{self.system_prompt}\n\n"
                f"=== DATA LITERATUR TEKS & TABEL EVALUASI ===\n"
                f"{context_text}\n\n"
                f"=== ASET VISUAL & DESKRIPSI GAMBAR KANDIDAT ===\n"
                f"{context_visual}\n\n"
                f"PERTANYAAN USER: {user_query}\n"
                f"JAWABAN:"
            )
            
            print("🧠 Menyintesis jawaban akhir menggunakan Llama3 lokal...")
            response = ollama.generate(model=self.model_name, prompt=full_prompt)
            
            end_time = time.time()
            final_answer = response['response']
            
            logger.info(f"Ollama sukses memproses jawaban dalam {end_time - start_time:.2f} detik.")
            return final_answer, self.current_trace
            
        except Exception as e:
            logger.error(f"Kegagalan sistem Ollama Agent: {e}")
            return f"Terjadi kesalahan fatal pada Ollama Agent: {str(e)}", []


# ============================================================================
# MAIN INTERACTION GATEWAY (Gerbang Eksekusi Pengujian Utama)
# ============================================================================
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    
    try:
        print("==========================================================")
        print(" Smart Intent Multimodal Ollama Gateway (Fixed Version)")
        print("==========================================================")
        
        # FUNGSI: Cukup hidupkan objek Agen saja, database sudah otomatis menyala di dalam __init__
        agent = ResearcherAgentOllama()
        
        print("\n✅ Sistem Siap! Silakan ketik pertanyaan Anda untuk menguji sensor.")
        print("(Ketik 'exit' atau 'keluar' untuk menghentikan pengujian)\n")
        
        while True:
            query_nyata = input("❓ Pertanyaan Uji Coba: ")
            
            if query_nyata.lower() in ['exit', 'keluar']:
                print("Menutup gerbang pengujian gateway.")
                break
                
            if not query_nyata.strip():
                continue
                
            # ▲ PERBAIKAN PANGGILAN: Cukup suapi 'query_nyata' saja, tidak usah mengirim objek retriever lagi.
            jawaban, jejak = agent.ask(query_nyata)
            
            print("\n----------------------------------------------------------")
            print("📊 JEJAK LOG BERPIKIR SENSOR AI:")
            for idx, step in enumerate(jejak):
                if step.get("step") == "SKIP_ACTION":
                    print(f"Langkah {idx+1} -> [DI-SKIP] Alat: '{step['tool_called']}' | Alasan: {step['reason']}")
                else:
                    print(f"Langkah {idx+1} -> [DIAKTIFKAN] Alat: '{step['tool_called']}' | Parameter: {step['arguments']}")
            print("----------------------------------------------------------")
            
            print("\n🤖 JAWABAN MULTIMODAL AI LOKAL:")
            print(jawaban)
            print("==========================================================\n")
            
    except Exception as e:
        print(f"❌ Gagal menjalankan pengujian gateway: {e}")
