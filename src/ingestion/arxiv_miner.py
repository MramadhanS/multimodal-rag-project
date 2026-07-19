# Library Import
import os
import json
import time
import random
import logging
from typing import List, Dict, Any
import arxiv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path Management
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Logging Configuration
logging.basicConfig (
    level=logging.INFO, 
    format = "%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "ingestion.log")),
              logging.StreamHandler()              
              ]
)

logger = logging.getLogger(__name__)

# Data Settings
RAW_PDF_DIR = os.path.join (BASE_DIR, "data", "raw_pdf")
METADATA_DIR = os.path.join (BASE_DIR, "data", "metadata")

# KEYWORD & CLUSTER TAXONOMY 
SEARCH_TAXONOMY = {
    "NLP_and_LLM": [
        "Large Language Models", "LLM Evaluation", "Retrieval-Augmented Generation", 
        "RAG Architecture", "Multimodal RAG", "Fine-Tuning", "PEFT", "LoRA", "QLoRA", 
        "Prompt Engineering", "Transformers", "Self-Attention Mechanism", "BERT", 
        "IndoBERT", "RoBERTa", "XLM-RoBERTa", "Text Embedding", "Vector Embeddings", 
        "Sentiment Analysis", "Named Entity Recognition", "NER", "Aspect-Based Sentiment Analysis", 
        "Text Summarization", "Sequence-to-Sequence"
    ],
    "Computer_Vision_and_Multimodal_AI": [
        "Object Detection", "YOLOv8", "YOLOv10", "YOLOv11", "Real-Time Object Detection", 
        "RT-DETR", "CornerNet", "CenterNet", "Vision Transformers", "ViT", "Swin Transformer", 
        "Contrastive Learning", "CLIP", "ImageBind", "Multimodal Embeddings", "Image Segmentation", 
        "Semantic Segmentation", "SAM", "Segment Anything Model", "Generative Adversarial Networks", 
        "GAN", "Diffusion Models", "Latent Diffusion", "Stable Diffusion"
    ],
    "MLOps_and_Data_Engineering": [
        "MLOps Pipeline", "Model Quantization", "LLM Optimization", "Vector Database", 
        "ANN Search", "Hybrid Search", "Knowledge Graphs", "Semantic Search", "Chunking Strategy"
    ]
}

# Arvix Data Miner
class ArxivDataMiner :
    def __init__(self, target_results: int = 300) :
        self.target_results = target_results
        # Konfigurasi client dengan delay 3 detik mematuhi aturan resmi arXiv
        self.client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

    def fetch_metadata (self) -> List[Dict[str,Any]] :
        master_metadata = [] 
        os.makedirs(METADATA_DIR, exist_ok=True)
        metadata_filepath = os.path.join(METADATA_DIR, "master_arxiv_metadata.jsonl")

        seen_ids = set()
        downloaded_count = 0

        # Loop Pencarian Berdasarkan Klaster Baru
        for category, queries in SEARCH_TAXONOMY.items() :
            for query in queries :
                if downloaded_count >= self.target_results:
                    break

                logger.info(f'Mencari Kategori: {category} | Kata Kunci: "{query}"')
                
                strict_query = f'(ti:"{query}" OR abs:"{query}") AND (cat:cs.LG OR cat:cs.CL OR cat:cs.CV)'
                
                search = arxiv.Search(
                    query=strict_query, 
                    max_results=30, 
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                try :
                    results_generator = self.client.results(search)
                    for result in results_generator:
                        if downloaded_count >= self.target_results:
                            break

                        current_id = result.get_short_id()
                        
                        # Filter 1: Hindari Duplikasi Paper ID
                        if current_id in seen_ids:
                            continue
                            
                        # Filter 2: Batasi 5 tahun terakhir (2021 hingga 2026)
                        pub_year = result.published.year
                        if pub_year < 2021:
                            continue 

                        seen_ids.add(current_id)

                        paper_data = {
                            "paper_id" : current_id,
                            "title" : result.title,
                            "abstract" : result.summary.replace("\n"," "),
                            "authors" : [a.name for a in result.authors],
                            "published_date" : result.published.strftime("%Y-%m-%d"),
                            "pdf_url" : result.pdf_url,
                            "domain_category" : category
                        }

                        master_metadata.append(paper_data)
                        downloaded_count += 1
                        logger.info(f" -> [{downloaded_count}/{self.target_results}] Terdata! | Tahun: {pub_year} | ID: {current_id}")

                except Exception as e :
                    logger.error(f"Gagal memproses kata kunci '{query}': {e}")
            
            if downloaded_count >= self.target_results:
                break

        logger.info(f"Pencarian selesai. Menulis {len(master_metadata)} data unik ke file JSONL...")
        with open (metadata_filepath, "w", encoding="utf-8") as f : 
            for paper in master_metadata:
                f.write (json.dumps(paper) + "\n")

        return master_metadata


    # PDF File Download (Private)
    def _download_pdf_worker (self, paper : Dict[str,Any], max_retries : int = 4) -> bool :
        os.makedirs (RAW_PDF_DIR, exist_ok=True)
        safe_id = paper["paper_id"].replace("/","_")
        filepath = os.path.join(RAW_PDF_DIR,f'{safe_id}.pdf')

        if os.path.exists(filepath) and os.path.getsize(filepath) > 102400:
            return True

        attempt = 0
        while attempt < max_retries :
            try :
                search = arxiv.Search(id_list = [paper["paper_id"]])
                result = next (arxiv.Client().results(search))
                result.download_pdf(dirpath=RAW_PDF_DIR, filename=f'{safe_id}.pdf')

                if os.path.exists(filepath) and os.path.getsize(filepath) > 102400:
                    time.sleep(random.uniform(4.0, 7.0)) 
                    return True
                else:
                    attempt += 1
                    time.sleep(5)

            except Exception as e:
                    attempt += 1
                    sleep_time = (3 ** attempt) + random.uniform(2, 5)
                    time.sleep(sleep_time)

        return False          

    # Parallel Download
    def download_pdfs_in_parallel(self, metadata_list: List[Dict[str, Any]], num_workers: int = 3):
        logger.info (f'Mulai Unduh Paralel ({num_workers} workers)')
        success_count = 0

        with ThreadPoolExecutor (max_workers=num_workers) as executor :
            futures = {executor.submit(self._download_pdf_worker, p) : p for p in metadata_list}
            for future in as_completed(futures) :
                if future.result() :
                    success_count += 1

        logger.info(f"Berhasil mengunduh {success_count}/{len(metadata_list)} PDF.")            


# =====================================================================
# UTAMA / EXECUTION GATE
# =====================================================================
if __name__ == "__main__" :
    metadata_filepath = os.path.join(METADATA_DIR, "master_arxiv_metadata.jsonl")
    
    if os.path.exists(metadata_filepath):
        try:
            os.remove(metadata_filepath)
            logger.info("Berhasil membersihkan sisa database metadata lama.")
        except Exception:
            pass

    miner = ArxivDataMiner(target_results=300)
    logger.info("Memulai pengambilan data 300 jurnal dari server arXiv...")
    metadata_list = miner.fetch_metadata()

    if metadata_list:
        logger.info(f"Mengirim {len(metadata_list)} target unik ke fungsi unduh paralel...")
        miner.download_pdfs_in_parallel(metadata_list, num_workers=1)
