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
        metadata_filepath = os.path.join (METADATA_DIR, "master_arxiv_metadata.jsonl")

        # Loop For Searching
        for category, queries in SEARCH_TAXONOMY.items() :
            for query in queries :
                logger.info (f'Category = {category} | Query = {query}')
                search = arxiv.Search(query=query,max_results=self.max_results,sort_by=arxiv.SortCriterion.SubmittedDate)

                try :
                    for result in self.client.results (search) :
                        paper_data = {
                            "paper_id" : result.get_short_id(),
                            "title" : result.title,
                            "abstract" : result.summary.replace("\n"," "),
                            "authors" : [a.name for a in result.authors],
                            "published_date" : result.published.strftime("%Y-%m-%d"),
                            "pdf_url" : result.pdf_url,
                            "domain_category" : category

                        }

                        master_metadata.append(paper_data)

                        with open (metadata_filepath,"a",encoding="utf-8") as f : 
                            f.write (json.dumps(paper_data) + "\n")

                    time.sleep(3)

                except Exception as e :
                    logger.error(f"API Key pada {query} : {e}")

        return master_metadata

    # PDF File Download (Private)
    def _download_pdf_worker (self, paper : Dict[str,Any], max_retries : int = 4) -> bool :
        os.makedirs (RAW_PDF_DIR, exist_ok=True)
        safe_id = paper["paper_id"].replace("/","_")
        filepath = os.path.join(RAW_PDF_DIR,f'{safe_id}.pdf')

        attempt = 0
        while attempt < max_retries :
            try :
                search = arxiv.Search(id_list = [paper["paper_id"]])
                result = next (arxiv.Client().results(search))
                result.download_pdf(dirpath=RAW_PDF_DIR, filename=f'{safe_id}.pdf')
                time.sleep(random.uniform(1.0,2.5))

                return True

            except HTTPError as e :
                if e.code == 429 :
                    attempt += 1
                    sleep_time = (2 ** attempt) + random.uniform(0,1)
                    logger.warning(f"Terkena Rate Limit (429). Mundur selama {sleep_time:.2f} detik...")
                    time.sleep (sleep_time)
                else :
                    break

            except Exception :
                break

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
    metadata_list = miner.fetch_metadata()

    if metadata_list :
        miner.download_pdfs_in_parallel(metadata_list,num_workers=3)