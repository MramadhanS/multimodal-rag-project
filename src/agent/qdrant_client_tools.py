# Import Library
import os
import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# Logger Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/agent_system.log"),
    level=logging.INFO,
    format="%(asctime)s | [QDRANT_RETRIEVER_DUAL] | %(message)s"
)
logger = logging.getLogger(__name__)

# Core Dual Retriever Class (Singleton Pattern)
class MultimodalQdrantRetriever:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Inisiasi MultimodalQdrantRetriever DUAL ENGINE (Online Stabil)...")
            print("🌐 Menghubungkan ke Jaringan Dual Model Hub...")
            
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            
            cls._instance = super(MultimodalQdrantRetriever, cls).__new__(cls)
            
            cls._instance.client = QdrantClient(host="localhost", port=6333)
            
            cls._instance.text_collection = "scientific_texts"
            cls._instance.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            cls._instance.image_collection = "scientific_images"
            cls._instance.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            cls._instance.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        return cls._instance


    # FUNCTION A: Berburu Teks dan Tabel Inline
    def search_text_and_tables(self, query: str, limit: int = 3) -> str:
        logger.info(f"Pencarian TEKS-TABEL untuk query: '{query}'")
        try:
            query_vector = self.text_model.encode(query).tolist()
            type_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="unified_page"))])
            
            search_results = self.client.search(
                collection_name=self.text_collection,
                query_vector=query_vector,
                query_filter=type_filter,
                limit=limit
            )
            if not search_results:
                return "OBSERVASI TEKS: Tidak ada literatur teks atau data metrik relevan yang ditemukan."
                
            contexts = []
            for hit in search_results:
                paper_id = hit.payload.get("paper_id", "Unknown")
                page_number = hit.payload.get("page_number", "Unknown")
                content = hit.payload.get("content", "")
                contexts.append(
                    f"--- BEGIN TEXT SOURCE [PAPER_ID: {paper_id} | PAGE: {page_number}] (Score: {hit.score:.2f}) ---\n"
                    f"{content}\n"
                    f"--- END TEXT SOURCE ---\n"
                )
            return "\n".join(contexts)
        except Exception as e:
            logger.error(f"Error search_text: {e}")
            return f"ERROR: Kegagalan akses database teks ({str(e)})."

    # FUNCTION B: Berburu Gambar Diagram / Grafik Menggunakan CLIP
    def search_images_via_clip(self, query: str, limit: int = 2) -> str:
        logger.info(f"Pencarian GAMBAR (CLIP) untuk query: '{query}'")
        try:
            # Merubah teks pertanyaan menjadi koordinat ruang gambar 512 dimensi
            inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
            text_features = self.clip_model.get_text_features(**inputs)
            query_vector = text_features.flatten().tolist()
            
            # Tembak koleksi scientific_images di Qdrant
            search_results = self.client.search(
                collection_name=self.image_collection,
                query_vector=query_vector,
                limit=limit
            )
            if not search_results:
                return "OBSERVASI VISUAL: Tidak ditemukan gambar atau diagram yang cocok."
                
            image_contexts = []
            for hit in search_results:
                paper_id = hit.payload.get("paper_id", "Unknown")
                caption = hit.payload.get("caption", "No Caption")
                file_path = hit.payload.get("file_path", "")
                image_contexts.append(
                    f"--- BEGIN VISUAL OBJECT SOURCE ---\n"
                    f"IMAGE_PATH: {file_path}\n"
                    f"CAPTION: {caption}\n"
                    f"SOURCE_PAPER: {paper_id}\n"
                    f"RELEVANCE_SCORE: {hit.score:.2f}\n"
                    f"--- END VISUAL OBJECT SOURCE ---"
                )
            return "\n\n".join(image_contexts)
        except Exception as e:
            logger.error(f"Error search_images: {e}")
            return f"ERROR: Kegagalan akses database gambar ({str(e)})."

if __name__ == "__main__":
    retriever = MultimodalQdrantRetriever()
    test_query = "Show me the pipeline flowchart diagram of multimodal RAG architecture"
    print(retriever.search_images_via_clip(query=test_query, limit=1))
