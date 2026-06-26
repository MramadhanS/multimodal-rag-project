# Library Import
import os
import json
import time
import random
import logging
from typing import List, Dict, Any
import arxiv
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

# Path Management
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR,exist_ok=True)

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

#Keyword
SEARCH_TAXONOMY = {
    "RAG": ["retrieval augmented generation", "multimodal RAG"],
    "LLM": ["large language model architecture", "Transformer attention"],
    "NLP_Local": ["Indonesian NLP", "cross lingual language model"]
}

# Arvix Data Miner
class ArxivDataMiner :
    def __init__(self, max_results : int = 50) :
        self.max_results = max_results
        self.client = arxiv.Client(page_size=50,delay_seconds=3.0,num_retries=3)

    def fetch_metadata (self) -> List[Dict[str,Any]] :
        master_metadata = [] 
        os.makedirs(METADATA_DIR, exist_ok=True)
        metadata_filepath = os.path.join(METADATA_DIR, "master_arxiv_metadata.jsonl")

        seen_ids = set()

        # Loop For Searching
        for category, queries in SEARCH_TAXONOMY.items() :
            for query in queries :
                logger.info (f'Category = {category} | Query = {query}')
                search = arxiv.Search(query=query, max_results=self.max_results, sort_by=arxiv.SortCriterion.SubmittedDate)

                try :
                    for result in self.client.results (search) :
                        current_id = result.get_short_id()
                        
                        if current_id in seen_ids:
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

                    time.sleep(3)

                except Exception as e :
                    logger.error(f"API Error pada {query} : {e}")

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
                    time.sleep(random.uniform(5.0, 9.0)) 
                    return True
                else:
                    logger.warning(f"File {safe_id}.pdf terputus/gagal disimpan. Mencoba kembali...")
                    attempt += 1
                    time.sleep(6)

            except Exception as e:
                    attempt += 1
                    sleep_time = (3 ** attempt) + random.uniform(2, 5)
                    logger.warning(f"Gangguan jaringan/Rate Limit. Mundur selama {sleep_time:.2f} detik... Error: {e}")
                    time.sleep(sleep_time)

        return False          

    # Parallel Download
    def download_pdfs_in_parallel(self, metadata_list: List[Dict[str, Any]], num_workers: int = 3):
        logger.info (f'Mulai Unduh Paralel ({num_workers} workers)')
        success_count = 0

        with ThreadPoolExecutor (max_workers=num_workers) as executor :
            futures = {executor.submit(self._download_pdf_worker,p) : p for p in metadata_list}
            for future in as_completed(futures) :
                if future.result() :
                    success_count += 1

        logger.info(f"Berhasil mengunduh {success_count}/{len(metadata_list)} PDF.")            


if __name__ == "__main__" :
    miner = ArxivDataMiner(max_results=20)
    metadata_filepath = os.path.join(METADATA_DIR, "master_arxiv_metadata.jsonl")
    metadata_list = []

    if os.path.exists(metadata_filepath) and os.path.getsize(metadata_filepath) > 0 :
        logger.info("File JSON Lokal Ditemukan, mulai memindai duplikat...")

        seen_ids = set()
        
        with open(metadata_filepath, 'r', encoding='utf-8') as f :
            for line in f :
                if line.strip() :
                    paper = json.loads(line.strip())
                    paper_id = paper.get("paper_id")
                    
                    if paper_id not in seen_ids:
                        seen_ids.add(paper_id)
                        metadata_list.append(paper)
        
        logger.info(f"Berhasil menyaring duplikat! Menemukan {len(metadata_list)} dokumen unik.")

        logger.info("Menulis ulang file master_arxiv_metadata.jsonl agar sinkron...")
        with open(metadata_filepath, "w", encoding="utf-8") as f_clean:
            for clean_paper in metadata_list:
                f_clean.write(json.dumps(clean_paper) + "\n")
                
        logger.info("File JSONL di harddisk berhasil diperbarui menjadi bersih!")

    else:
        logger.info("File JSONL lokal tidak ditemukan/kosong. Menghubungi server arXiv...")
        metadata_list = miner.fetch_metadata()

    if metadata_list:
        logger.info(f"Mengirim {len(metadata_list)} target unik ke fungsi download...")
        miner.download_pdfs_in_parallel(metadata_list, num_workers=1)
