# Import Library
import fitz
import os
import json
import logging
import glob

# Directory & Logging Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename= os.path.join(BASE_DIR,"logs/extraction.log"), # Diubah menjadi ekstensi .log standar
    level= logging.INFO,
    format= "%(asctime)s | [IMAGE_EXTRACTOR] | %(message)s"
)

logger = logging.getLogger(__name__)

RAW_PDF_DIR = os.path.join(BASE_DIR,"data/raw_pdf")
IMAGE_OUT_DIR = os.path.join(BASE_DIR,"data/processed/chunks_images") # Ditambahkan data/ agar struktur folder rapi

os.makedirs(IMAGE_OUT_DIR,exist_ok=True)

# Image Extractor
class AcademicImageExtractor :
    def __init__ (self, min_image_size_bytes : int = 15360) :
        self.min_image_size_bytes = min_image_size_bytes

    # Contextual Caption Extraction Heuristic
       # Contextual Caption Extraction Heuristic
    def _extract_caption_heuristic(self, page : fitz.Page, img_bbox : fitz.Rect) -> str :
        text_blocks = page.get_text("blocks")
        candidates = []
        caption_keywords =  ("figure", "fig.", "fig ", "gambar", "architecture", "framework", "pipeline", "diagram", "table", "tabel")

        # Scan & Filter Valid Caption Blocks
        for block in text_blocks :
            bx0, by0, bx1, by1, block_text, _, _ =  block
            cleaned_text = block_text.replace("\n"," ").strip()

            if not cleaned_text :
                continue

            if cleaned_text.lower().startswith(caption_keywords) :
                if bx1 < (img_bbox.x0 - 50) or bx0 > (img_bbox.x1 + 50):
                    continue

                distance_to_img = min(abs(by0 - img_bbox.y1), abs(by1 - img_bbox.y0))

                candidates.append ({
                    "text" : cleaned_text, 
                    "distance" : distance_to_img,
                    "position" : "below" if by0 >= img_bbox.y1 else "above"
                })

        # Select & Process Best Caption Candidate
        if candidates :
            candidates.sort(key=lambda x : x["distance"])
            best_candidates = candidates[0]

            if best_candidates["distance"] < 350 :
                final_caption = best_candidates["text"]

                if len(final_caption) > 1000 :
                    cutoff = final_caption[:1000].rfind(".")
                    if cutoff > 100 :
                        final_caption = final_caption[:cutoff+1]
                    else :
                        final_caption = final_caption [:1000] + "..."

                return final_caption.strip()
        
        # Fallback Spatial Multi-Directional Search
        extended_rect = fitz.Rect (
            img_bbox.x0 - 40,
            max(0, img_bbox.y0 - 150), 
            img_bbox.x1 + 40,
            img_bbox.y1 + 250 
        )       

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
                if line.endswith((".", ". ")): 
                    break
            if len(fallback_caption) > 400:
                break
                
        if fallback_caption:
            return fallback_caption.strip()

        return "Visual Object (No Caption Found)"

    
    #  Core PDF Extraction Pipeline
    def process_pdf(self, pdf_path: str):
        paper_id = os.path.basename(pdf_path).replace(".pdf", "")
        paper_img_dir = os.path.join(IMAGE_OUT_DIR, paper_id)
        os.makedirs(paper_img_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            metadata_list = []
            extracted_count = 0
            
            # Loop Pages & Extract Image Streams
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True) 
                
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"] 

                    if len(image_bytes) < 2048: 
                        continue

                    rects = page.get_image_rects(xref)
                    if not rects: 
                        continue
                    img_bbox = rects[0]

                    caption = self._extract_caption_heuristic(page, img_bbox)

                    filename = f"page_{page_num+1}_img_{img_idx+1}.{ext}"
                    filepath = os.path.join(paper_img_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    metadata_list.append({
                        "paper_id": paper_id, 
                        "page_number": page_num + 1,
                        "image_path": filepath,
                        "caption": caption
                    })
                    extracted_count += 1

            if metadata_list:
                with open(os.path.join(paper_img_dir, "_image_metadata.json"), "w") as f:
                    json.dump(metadata_list, f, indent=2)
                
                logger.info(f"[{paper_id}] Berhasil mengekstrak {extracted_count} gambar asli.")
            
        except Exception as e:
            logger.error(f"Gagal proses gambar pada {pdf_path}: {e}")

if __name__ == "__main__":
    extractor = AcademicImageExtractor(min_image_size_bytes=2048)
    for pdf in glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf")):
        extractor.process_pdf(pdf)
