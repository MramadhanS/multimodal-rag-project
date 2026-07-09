# Import Library
import os
import glob
import uuid
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Logger & Directory Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename = os.path.join(BASE_DIR, "logs/vector_db.log"),
    level= logging.INFO,
    format="%(asctime)s | [TEXT_INDEXER] | %(message)s"
)
logger = logging.getLogger(__name__)

TEXT_DIR = os.path.join(BASE_DIR, "data/processed/chunks_text")

# Core Class Function
class TextTableIndexer :

    # FUNCTION: Initialize Configurations
    def __init__ (self) :
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "scientific_texts"
        logger.info("memuat model MPNet ke RAM")
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self._ensure_collection_exists()

    # FUNCTION: Collection Validator Gateway
    def _ensure_collection_exists (self) :
        try :
            self.client.get_collection(self.collection_name)
            logger.info (f'Collection {self.collection_name} aktif')
        except Exception :
            logger.info (f'Mebuat Collection baru : {self.collection_name}')
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config = VectorParams(size = self.vector_size, distance = Distance.COSINE)
            )

    # Memory Batching Generator
    def _process_batches (self, items : List[Any], batch_size : int = 32) : 
        for i in range (0, len(items), batch_size) :
            yield items [i:i + batch_size]

    # Vector Transformation and Upsert Pipeline
    def _upload_to_qdrant (self, payloads : List[Dict[str, Any]], doc_type : str) :
        if not payloads :
            return
        
        batch_limit = 32
        total_batches = (len(payloads) // batch_limit) + 1

        for idx, batch in enumerate (self._process_batches(payloads, batch_limit)) :
             logger.info(f"Encoding & Uploading {doc_type} Batch {idx+1}/{total_batches}...")

             texts = [item["content"] for item in batch]
             vectors = self.model.encode(texts, show_progress_bar=False).tolist()

             points = [
                 PointStruct(
                     id=str(uuid.uuid4()),
                     vector=vectors[i],
                     payload=batch[i]
                 ) for i in range (len(batch))
             ]

             self.client.upsert (collection_name=self.collection_name, points=points)

    # FUNCTION: Deep Document Repository Scanner
    def index_unified_documents (self) :
        logger.info ("Memulai Pemindaian Files")
        payloads = []

        md_files = glob.glob (os.path.join(TEXT_DIR, "**/*.md"), recursive=True)
        logger.info(f"Ditemukan {len(md_files)} file halaman Markdown untuk dimasukkan ke database.")

        for md_file in md_files:
            paper_id = os.path.basename(os.path.dirname(md_file))
            page_num_match = re.search(r'page_(\d+)', os.path.basename(md_file))
            page_num = int(page_num_match.group(1)) if page_num_match else 0
        
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if not content.strip():
                continue

            payloads.append({
                "doc_type": "unified_page",
                "paper_id": paper_id,
                "page_number": page_num,
                "content": content
            })
            
        self._upload_to_qdrant(payloads, "INTEGRATED_TEXT_TABLE")
        logger.info(f"Proses indeks selesai. Total {len(payloads)} halaman sukses ditanam di Qdrant.")

# MAIN GATEWAY 
if __name__ == "__main__":
    import re 
    indexer = TextTableIndexer()
    indexer.index_unified_documents()
