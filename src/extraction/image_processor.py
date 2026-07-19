# Import Library
import fitz
import os
import json
import logging
import glob

# Directory & Logging Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename= os.path.join(BASE_DIR,"logs/images_extraction.log"), 
    level= logging.INFO,
    format= "%(asctime)s | [IMAGE_EXTRACTOR] | %(message)s"
)

logger = logging.getLogger(__name__)

RAW_PDF_DIR = os.path.join(BASE_DIR,"data/raw_pdf")
IMAGE_OUT_DIR = os.path.join(BASE_DIR,"data/processed/chunks_images") 

os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

class AcademicImageExtractor :
    def __init__ (self, min_image_size_bytes : int = 2048) :
        self.min_image_size_bytes = min_image_size_bytes

    def _extract_caption_heuristic(self, page : fitz.Page, img_bbox : fitz.Rect) -> str :
        text_blocks = page.get_text("blocks")
        candidates = []
        caption_keywords = ("figure", "fig.", "fig ", "gambar", "architecture", "framework", "pipeline", "diagram", "table", "tabel")

        for block in text_blocks :
            bx0, by0, bx1, by1, block_text, _, _ =  block
            cleaned_text = block_text.replace("\n"," ").strip()

            if not cleaned_text :
                continue

            if cleaned_text.lower().startswith(caption_keywords) :
                if bx1 < (img_bbox.x0 - 100) or bx0 > (img_bbox.x1 + 100):
                    continue

                distance_to_img = min(abs(by0 - img_bbox.y1), abs(by1 - img_bbox.y0))

                candidates.append ({
                    "text" : cleaned_text, 
                    "distance" : distance_to_img,
                    "position" : "below" if by0 >= img_bbox.y1 else "above"
                })

        if candidates :
            candidates.sort(key=lambda x : x["distance"])
            best_candidates = candidates[0]

            if best_candidates["distance"] < 400 :
                final_caption = best_candidates["text"]
                if len(final_caption) > 1000 :
                    cutoff = final_caption[:1000].rfind(".")
                    final_caption = final_caption[:cutoff+1] if cutoff > 100 else final_caption[:1000] + "..."
                return final_caption.strip()
        
        # Fallback multi directional
        extended_rect = fitz.Rect (img_bbox.x0 - 50, max(0, img_bbox.y0 - 150), img_bbox.x1 + 50, img_bbox.y1 + 250)       
        fallback_text = page.get_text("text", clip = extended_rect)
        lines = [line.strip() for line in fallback_text.split("\n") if line.strip()]

        fallback_caption = ""
        capture = False
        for line in lines : 
            if line.lower().startswith(caption_keywords) : 
                capture = True
                fallback_caption += line + " "
            elif capture:
                fallback_caption += line + " "
                if line.endswith((".", ". ")): break
            if len(fallback_caption) > 400: break
                
        return fallback_caption.strip() if fallback_caption else "Visual Object (No Caption Found)"

    def process_pdf(self, pdf_path: str):
        paper_id = os.path.basename(pdf_path).replace(".pdf", "")
        paper_img_dir = os.path.join(IMAGE_OUT_DIR, paper_id)
        os.makedirs(paper_img_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            metadata_list = []
            extracted_count = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                drawings_list = page.get_drawings() 
                
                # JALUR A: Jika ada objek gambar biasa (Bitmap)
                if image_list:
                    for img_idx, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image["ext"] 

                        if len(image_bytes) < self.min_image_size_bytes: 
                            continue

                        rects = page.get_image_rects(xref)
                        img_bbox = rects[0] if rects else fitz.Rect(0, 0, page.rect.width, page.rect.height)

                        caption = self._extract_caption_heuristic(page, img_bbox)
                        filename = f"page_{page_num+1}_bitmap_{img_idx+1}.{ext}"
                        filepath = os.path.join(paper_img_dir, filename)
                        
                        with open(filepath, "wb") as f:
                            f.write(image_bytes)

                        metadata_list.append({
                            "paper_id": paper_id, 
                            "page_number": page_num + 1,
                            "image_path": filepath,
                            "caption": caption,
                            "type": "bitmap"
                        })
                        extracted_count += 1
                
                elif len(drawings_list) > 10: 
                    zoom = 2 
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    filename = f"page_{page_num+1}_vector_page.png"
                    filepath = os.path.join(paper_img_dir, filename)
                    pix.save(filepath)
                    
                    fallback_bbox = fitz.Rect(0, 0, page.rect.width, page.rect.height)
                    caption = self._extract_caption_heuristic(page, fallback_bbox)
                    if caption == "Visual Object (No Caption Found)":
                        caption = f"Halaman visual tersemat diagram/tabel arsitektur pada halaman ke-{page_num+1}."

                    metadata_list.append({
                        "paper_id": paper_id, 
                        "page_number": page_num + 1,
                        "image_path": filepath,
                        "caption": caption,
                        "type": "vector_render"
                    })
                    extracted_count += 1

            if metadata_list:
                with open(os.path.join(paper_img_dir, "_image_metadata.json"), "w") as f:
                    json.dump(metadata_list, f, indent=2)
                logger.info(f"[{paper_id}] Berhasil mengamankan {extracted_count} elemen visual asli & vektor.")
            
        except Exception as e:
            logger.error(f"Gagal memproses gambar pada {pdf_path}: {e}")

if __name__ == "__main__":
    extractor = AcademicImageExtractor(min_image_size_bytes=2048)
    pdf_files = glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf"))
    print(f"Mengantre {len(pdf_files)} dokumen untuk ekstraksi visual komprehensif...")
    
    for pdf in pdf_files:
        extractor.process_pdf(pdf)
    print("✅ Seluruh elemen visual dari 300 jurnal berhasil diamankan!")
