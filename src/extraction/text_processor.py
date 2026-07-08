# Import Library
import os
import glob
import logging
import re
import pymupdf4llm

# Configuration Logger & Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename=os.path.join(BASE_DIR,"logs/text_extraction.log")
    level=logging.INFO,
    format="%(asctime)s | [TEXT_PROCESSOR_ULTIMATUM] | %(message)s"
)

logger = logging.getLogger(__name__)

# Directory File
RAW_PDF_DIR = os.path.join(BASE_DIR,"data/raw_pdf")
TEXT_OUT_DIR = os.path.join(BASE_DIR,"data/processed/chunks_text")

os.makedirs(TEXT_OUT_DIR,exist_ok=True)

# Text Exctractor Class Function
class AcademicTextProcessor :
    def __init__(self):
        pass
    
    # Advanced Text Processing Filter
    def _post_clean_markdown (self,text_md : str) -> str :

        text_md = re.sub(r'-\s*<br\s*/?>\s*', '', text_md)
        text_md = re.sub(r'<br\s*/?>', ' ', text_md)
        text_md = re.sub(r'[ \t]+', ' ', text_md)
        
        return text_md.strip()
    
    # Core Function Processing PDF into MD File Text
    def process_pdf (self, pdf_path : str) :
        
        # Directory Configuration
        paper_id = os.path.basename(pdf_path).replace(".pdf","") 
        paper_text_dir = os.path.join(TEXT_OUT_DIR,paper_id)
        os.makedirs(paper_text_dir,exist_ok=True)

        # Extract Document into Chunks
        try :   
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks = True)
            saved_pages = 0

            # Iterate and Structure Every Single Page Context
            for chunk in md_pages:
            
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", metadata.get("page_number", saved_pages + 1))
                raw_content = chunk.get("text", "")
                clean_content = self._post_clean_markdown(raw_content)
            
                if not clean_content:
                    continue 
                    
                page_filepath = os.path.join(paper_text_dir, f"page_{page_num}.md")
                
                with open(page_filepath, "w", encoding="utf-8") as f:
                 
                    f.write(f"--- METADATA SCOPE: PAPER ID {paper_id} | ORIGINAL PAGE {page_num} ---\n\n")
                    f.write(clean_content)
                    
                saved_pages += 1
            
            logger.info(f"[{paper_id}] Sukses mengamankan {saved_pages} halaman teks dan tabel terintegrasi.")
            
        except Exception as e:
            logger.error(f"Gagal total memproses teks terpadu pada {paper_id}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Integrated Academic Text Extraction...")
    extractor = AcademicTextProcessor()
    
    pdf_queue = glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf"))
    
    if not pdf_queue:
        logger.warning("Tidak ditemukan file PDF di folder data/raw_pdf/!")
    else:
        logger.info(f"Ditemukan {len(pdf_queue)} file PDF siap diproses.")
        for pdf in pdf_queue:
            extractor.process_pdf(pdf)
            
    logger.info("Integrated Extraction Process Finished Successfully.")


