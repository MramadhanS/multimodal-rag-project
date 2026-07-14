# Import Library
import os
import time
import json
import mlflow
import logging
from pathlib import Path

# Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logger = logging.getLogger(__name__)

# Konfigurasi MLflow untuk menyimpan log di folder lokal
mlflow.set_tracking_uri(Path(BASE_DIR) / "mlruns")
mlflow.set_experiment("Agentic_RAG_Evaluations")

# MLFlow Function
def log_agent_interaction (query : str, answer : str, latency : float, trace : list) :
    try :
        with mlflow.start_run(run_name=f'Chat_{int(time.time())}') :
            mlflow.log_param("user_query",query)
            answer_path = "temp_answer.txt"

            with open(answer_path,"w",encoding="utf-8") as f :
                f.write(answer)

            mlflow.log_artifact(answer_path,"outputs")
            os.remove(answer_path)

            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("tools_called_count", len(trace))

            trace_path = "temp_trace.json"
            
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)
                
            mlflow.log_artifact(trace_path, "reasoning_traces")
            os.remove(trace_path) 

        logger.info("Berhasil mencatat interaksi ke MLflow.")
        
    except Exception as e:
        logger.error(f"Gagal mencatat ke MLflow: {e}")   


   