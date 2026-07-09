# Import Library
import os
import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Logger Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/agent_system.log"),
    level=logging.INFO,
    format="%(asctime)s | [QDRANT_TOOL_ULTIMATUM] | %(message)s"
)
logger = logging.getLogger(__name__)

class TextTableRetriever :
    _instance = None

    # FUNCTION: Control Singleton Instance Initialization
    def __new__ (cls) :
        if cls._instance is None :
            logger.info("Inisiasi TextTableRetriever dan memuat model embedding...")
            cls._instance = super(TextTableRetriever,cls).__new__(cls)
            cls._instance.client = QdrantClient(host='localhost',port=6333)
            cls._instance.collection_name = "scientific_texts"
            cls._instance.model = SentenceTransformer('all-MiniLM-L6-v2')

        return cls._instance
    
    # FUNCTION: Hybrid Context Retrieval Vector Search
    def search_qdrant(self, query: str, limit: int = 3) -> str:
       
        logger.info(f"Mencari dokumen semantik terpadu untuk query: '{query}'")

        try :
            query_vector = self.model.encode(query).tolist()
            type_filter = Filter(
                must = [FieldCondition(key='doc_type',match=MatchValue(value='unified_page'))]
            )
            
            search_results = self.client.search(
                collection_name = self.collection_name,
                query_vector = query_vector, 
                query_filter = type_filter,
                limit = limit
            )
        
            if not search_results :
                 return "OBSERVASI: Tidak ada literatur atau data metrik relevan yang ditemukan untuk kueri ini."
            
            contexts = []

            for hit in search_results:
                
                paper_id = hit.payload.get("paper_id", "Unknown")
                page_number = hit.payload.get("page_number", "Unknown")
                content = hit.payload.get("content", "")
                score = hit.score # Nilai kedekatan semantik (0.00 sampai 1.00)
                
                contexts.append(
                    f"--- BEGIN SOURCE [PAPER_ID: {paper_id} | PAGE: {page_number}] (Relevance Score: {score:.2f}) ---\n"
                    f"{content}\n"
                    f"--- END SOURCE [PAPER_ID: {paper_id} | PAGE: {page_number}] ---\n"
                )

            return "\n".join(contexts)
        
        except Exception as e :
            logger.error(f"Error saat query ke Qdrant: {e}")
            return f"SYSTEM ERROR: Kegagalan akses database ({str(e)})."

# MAIN GATEAWAY
if __name__ == "__main__":
    retriever = TextTableRetriever()

    test_query = "How does the multimodal RAG framework handle visual tokens and cross-view geometric consistency before feeding into the LLM?"
    print(f"🔍 Menguji pencarian semantik untuk: '{test_query}'...\n")
    
    raw_response = retriever.search_qdrant(query=test_query, limit=2)
    print(raw_response)