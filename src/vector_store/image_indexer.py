# Import Library
import os
import json
import glob
import uuid
import logging
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Logger & Directory Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/visual_db.log"),
    level=logging.INFO,
    format="%(asctime)s | [IMAGE_INDEXER_CLIP] | %(message)s"
)
logger = logging.getLogger(__name__)

IMAGE_DIR = os.path.join(BASE_DIR, "data/processed/chunks_images")

class ImageVectorIndexer:

    # FUNCTION: Initialize Configurations and Heavy-Duty AI Engine
    def __init__(self):
        
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "scientific_images"
        
        logger.info("Memuat model OpenAI CLIP (ViT-B/32) ke RAM/VRAM...")
        print("🤖 Memuat model OpenAI CLIP Vision AI...")
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vector_size = 512 
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
       
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' terdeteksi aktif.")
        except Exception:
            logger.info(f"Membuat collection gambar baru: '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
    
    def embed_image (self, image_path : str) -> list :

        try :
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors='pt')
            image_features = self.model.get_image_features(**inputs)

            return image_features.flatten().tolist()
        
        except Exception as e:
            logger.error(f"Gagal memproses gambar {image_path}: {e}")
            return []
        
    def index_all_images(self):
    
        logger.info("Mulai memproses Vektor Gambar...")
        print("📁 Memulai pemindaian direktori gambar...")
      
        if not os.path.exists(IMAGE_DIR):
            print("❌ ERROR: Folder data/processed/chunks_images tidak ditemukan!")
            return
    
        paper_folders = os.listdir(IMAGE_DIR)
        print(f"📄 Menemukan {len(paper_folders)} folder dokumen riset visual.")

        # Prosessing Journal Images
        for paper_id in paper_folders:
            paper_dir = os.path.join(IMAGE_DIR, paper_id)
           
            if not os.path.isdir(paper_dir): 
                continue

            metadata_file = os.path.join(paper_dir, "_image_metadata.json")
            
            if not os.path.exists(metadata_file): 
                continue
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                images_metadata = json.load(f)
                
            points = []
            print(f"📸 Encoding aset visual untuk Paper: {paper_id}...")
            
            for img_meta in images_metadata:
                image_path = img_meta["image_path"]
                if not os.path.exists(image_path): 
                    continue
                
                vector = self.embed_image(image_path)
            
                if not vector: 
                    continue

                clean_universal_path = Path(image_path).as_posix()
                
                payload = {
                    "doc_type": "image",
                    "paper_id": img_meta["paper_id"],
                    "caption": img_meta["caption"],
                    "file_path": clean_universal_path
                }
                
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=vector,      
                        payload=payload       
                    )
                )
            
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info(f"[{paper_id}] Berhasil meng-index {len(points)} vektor gambar ke Qdrant.")
                
        print("✅ PROSES INDEKS GAMBAR SELESAI 100%! Silakan cek dashboard Qdrant Anda.")

# MAIN GATEWAY 
if __name__ == "__main__":
    indexer = ImageVectorIndexer()
    indexer.index_all_images()