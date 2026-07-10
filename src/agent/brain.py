# Import Library
import os
import logging
import time
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import content_types

load_dotenv()

from src.agent.agent_tools import AVAILABLE_TOOLS

# Directory & Logging Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/agent_system.log"),
    level=logging.INFO,
    format="%(asctime)s | [AGENT_BRAIN_ULTIMATUM] | %(message)s"
)
logger = logging.getLogger(__name__)


# Core Class Function
class ResearcherAgent :

    # API Key
    def __init__(self) :
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("CRITICAL: GEMINI_API_KEY tidak ditemukan di file .env atau OS Environment!")
            raise ValueError(
                "CRITICAL ERROR: GEMINI_API_KEY tidak ditemukan!\n"
                "Silakan cek kembali apakah Anda sudah membuat file '.env' di folder utama proyek Anda "
                "dan mengisinya dengan format: GEMINI_API_KEY=AIzaSy..."
            )
        
        genai.configure(api_key=self.api_key)

        self.system_instruction = """
Anda adalah AI Peneliti Literatur Ilmiah tingkat doktoral yang sangat objektif dan jujur. 
Tugas Anda adalah memberikan jawaban komprehensif, presisi, dan akademis berdasarkan jurnal penelitian.

Anda memiliki akses ke Vector Database yang berisi teks dan tabel metrik dari literatur AI terbaru.

PROTOKOL OPERASI STANDAR (SOP) KETAT:
1. NO HALLUCINATION: JANGAN PERNAH menjawab dari ingatan parametrik Anda sendiri mengenai data faktual, angka metrik, atau spesifikasi paper. 
2. GUNAKAN ALAT (TOOLS): Jika Anda butuh informasi untuk menjawab, panggil alat pencari yang disediakan. Jangan menebak.
3. MULTI-HOP REASONING: Jika pertanyaan sangat kompleks, panggil alat pencari, lihat hasilnya, lalu formulasikan kueri baru jika informasi belum lengkap.
4. KEWAJIBAN SITASI: Anda WAJIB menyertakan nomor identitas paper (contoh: [PAPER ID: 2606.23943v1]) di akhir setiap klaim, angka benchmark, atau kutipan teori. Dapatkan ID ini dari teks hasil kembalian Tools.
5. PENANGANAN KEKOSONGAN: Jika setelah menggunakan Tools Anda tidak menemukan informasi yang diminta, katakan dengan jujur: "Data tersebut belum tersedia di dalam database literatur saya." Jangan pernah mengarang data.
6. FORMATTING: Jika mengutip matriks evaluasi tabel, pertahankan format tulisan pipa Markdown (|---|---|) agar terender rapi di layar pengguna.
        """
        logger.info("Menginisialisasi Gemini 3.5 Flash dengan Function Calling terpadu...")
        self.model = genai.GenerativeModel(
            model_name='gemini-3.5-flash',
            system_instruction=self.system_instruction,
            tools=AVAILABLE_TOOLS 
        )
        
        self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)
        self.current_trace = []

        # Main Interaction Gateaway
    def ask (self, user_query : str) -> Tuple[str,List[Dict[str,Any]]] :
        """
        Menerima input pertanyaan string dari pengguna.
        Mengembalikan tuple: (Jawaban Komprehensif AI, Riwayat Proses Berpikir Trace)
        """
        logger.info(f"Menerima pertanyaan pengguna: '{user_query}'")
        self.current_trace = [] 

        try :
            time.sleep(3)

            start_time = time.time()
            response = self.chat_session.send_message(user_query)
            end_time = time.time()

            final_answer = response.text

            self._extract_reasoning_trace()

            logger.info(f"Agent sukses mensintesis jawaban dalam {end_time - start_time:.2f} detik")
            return final_answer, self.current_trace

        except Exception as e :
            logger.error(f"Kegagalan Sistem Agent AI : {e}")
            return f"Terjadi Kesalahan pada Agent AI : {str(e)}" , []
    
    # Telemetry Interception
    def _extract_reasoning_trace (self) :
        recent_history = self.chat_session.history[-10:]

        for message in recent_history : 
            if message.role == "model" and getattr(message,"parts",None) :
                for part in message.parts :
                    if getattr(part,"function_call", None) :
                        fc = part.function_call
                        tool_name = fc.name
                        args_dict = dict(fc.args)

                        trace_log = {
                            "step": "ACTION",
                            "tool_called": tool_name,
                            "arguments": args_dict
                        }

                        if trace_log not in self.current_trace:
                            self.current_trace.append(trace_log)
                            logger.info(f"TRACE RECORDED: Agen memanggil '{tool_name}' dengan parameter '{args_dict}'")

# Main Gateaway
if __name__ == "__main__":
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        agent = ResearcherAgent()
        print("\n======================================================================")
        print("🚀 GENUINE MULTIMODAL AGENT BRAIN")
        print("======================================================================\n")
        
        query = "Jelaskan bagaimana arsitektur RAG bekerja menurut paper terbaru di database saya."
        print(f"💬 USER QUESTION: {query}\n")
        
        print("🤖 Agen sedang berpikir, memilih alat, dan meracik analisis...")
        answer, trace = agent.ask(query)
        
        print("\n--- 🧠 PROSES BERPIKIR AGEN (REASONING TRACE) ---")
        if not trace:
            print("- Agen menjawab langsung menggunakan memori percakapan (No Tool Called).")
        for step in trace:
            print(f"🛠️  Mengeksekusi Alat Database : {step['tool_called']}")
            print(f"📌 Parameter Pencarian Kueri  : {step['arguments']}")
            print("-" * 50)
            
        print("\n--- ✨ JAWABAN FINAL SINTESIS AGEN AI ---")
        print(answer)
        
    except Exception as e:
        print(f"\n❌ Gagal inisiasi koordinat sistem saraf Agen. Error:\n{e}")