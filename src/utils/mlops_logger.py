import os
import time
import json
import mlflow
import logging
from pathlib import Path  

# Konfigurasi logger proyek internal
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logger = logging.getLogger(__name__)

# Mengunci alamat folder 'mlruns' lokal agar sinkron di Windows
mlflow.set_tracking_uri(Path(BASE_DIR) / "mlruns")
mlflow.set_experiment("Agentic_RAG_Evaluations")

def log_agent_interaction(query: str, answer: str, trace: list, latency: float):
    """
    Menyimpan setiap interaksi pengguna ke dalam dashboard MLflow secara langsung via memori.
    100% Bebas Eror File Locking Windows (Anti-Silang Merah ❌).
    """
    try:
        # Membuka sesi perekaman baru di MLflow berdasarkan timestamp detik saat ini
        with mlflow.start_run(run_name=f"Chat_{int(time.time())}"):
            
            # 1. Catat Parameter Input Teks Kueri
            mlflow.log_param("user_query", query)
            
            # 2. Catat Jawaban Panjang Langsung via RAM
            mlflow.log_text(answer, artifact_file="outputs/final_answer.txt")
            
            # 3. Catat Metrik Angka Statistik Kecepatan & Jumlah Perangkat
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("tools_called_count", len(trace))
            
            # 4. Catat Jejak Berpikir JSON Langsung via RAM
            trace_json_string = json.dumps(trace, indent=2)
            mlflow.log_text(trace_json_string, artifact_file="reasoning_traces/trace.json")
            
        logger.info("Berhasil mencatat interaksi ke MLflow (Status: FINISHED).")
        
    except Exception as e:
        logger.error(f"Gagal mencatat telemetri ke MLflow: {e}")
