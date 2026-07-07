# ============================================================================
# SECTION 1: IMPOR MODUL DAN LIBRARY
# ============================================================================
import os
import glob
import logging
import re
import pymupdf4llm

# ============================================================================
# SECTION 2: KONFIGURASI DIREKTORI & SISTEM LOGGING
# ============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs/text_extraction.log"),
    level=logging.INFO,
    format="%(asctime)s | [TEXT_PROCESSOR_ULTIMATUM] | %(message)s"
)
logger = logging.getLogger(__name__)

RAW_PDF_DIR = os.path.join(BASE_DIR, "data/raw_pdf")
TEXT_OUT_DIR = os.path.join(BASE_DIR, "data/processed/chunks_text")
os.makedirs(TEXT_OUT_DIR, exist_ok=True)

# ============================================================================
# SECTION 3: STRUKTUR UTAMA CORE UNIFIED PROCESSOR (OOP)
# ============================================================================
class AcademicTextProcessor:
    def __init__(self):
        pass

    def _post_clean_markdown(self, text_md: str) -> str:
        """
        STRATEGY 1: Advanced Text Post-Processing Filter.
        Membersihkan kode-kode sisa render visual PDF agar teks menjadi sangat
        nyaman dan bersih saat dibaca oleh LLM Agent RAG Anda nanti.
        """
        # 1. Bersihkan tanda hubung kata akibat patah baris otomatis (stan-<br>dard -> standard)
        text_md = re.sub(r'-\s*<br\s*/?>\s*', '', text_md)
        
        # 2. Ganti sisa tag HTML <br> biasa dengan spasi bersih
        text_md = re.sub(r'<br\s*/?>', ' ', text_md)
        
        # 3. Rapikan spasi ganda yang berlebih agar menghemat kuota Token API
        text_md = re.sub(r'[ \t]+', ' ', text_md)
        
        return text_md.strip()

    def process_pdf(self, pdf_path: str):
        """
        FUNCTION: Core Integrated Extraction Pipeline.
        Mengekstrak satu file PDF utuh halaman demi halaman secara holistik.
        Tabel dibiarkan mengalir di dalam teks asli (Inline Table).
        """
        paper_id = os.path.basename(pdf_path).replace('.pdf', '')
        paper_text_dir = os.path.join(TEXT_OUT_DIR, paper_id)
        os.makedirs(paper_text_dir, exist_ok=True)
        
        try:
            # STRATEGY 2: Extract Document with Granular Page-Chunking Enabled
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            
            saved_pages = 0
            
            # STRATEGY 3: Iterate and Structure Every Single Page Context
            for chunk in md_pages:
                # ▲ KUNCI FIX: Menggunakan .get() bertingkat untuk mengantisipasi perbedaan versi library
                # Mengecek kata kunci 'page' terlebih dahulu, jika zonk, otomatis beralih ke 'page_number'
                metadata = chunk.get("metadata", {})
                page_num = metadata.get("page", metadata.get("page_number", saved_pages + 1))
                
                raw_content = chunk.get("text", "")
                
                # Picu fungsi pembersihan teks kualitatif dan tag HTML sisa
                clean_content = self._post_clean_markdown(raw_content)
                
                if not clean_content:
                    continue # Abaikan halaman jika kosong tanpa teks/informasi biner
                    
                # Menyusun file output final berformat Markdown per halaman
                page_filepath = os.path.join(paper_text_dir, f"page_{page_num}.md")
                
                with open(page_filepath, "w", encoding="utf-8") as f:
                    # STRATEGY 4: Inject Tracking Metadata Header
                    f.write(f"--- METADATA SCOPE: PAPER ID {paper_id} | ORIGINAL PAGE {page_num} ---\n\n")
                    f.write(clean_content)
                    
                saved_pages += 1
                
            logger.info(f"[{paper_id}] Sukses mengamankan {saved_pages} halaman teks dan tabel terintegrasi.")
            
        except Exception as e:
            logger.error(f"Gagal total memproses teks terpadu pada {paper_id}: {str(e)}")


# ============================================================================
# SECTION 4: GERBANG EKSEKUSI UTAMA
# ============================================================================
if __name__ == "__main__":
    logger.info("Starting Integrated Academic Text Extraction...")
    extractor = AcademicTextProcessor()
    
    # Mencari dan mengantrekan seluruh file PDF yang ada di dalam folder raw_pdf
    pdf_queue = glob.glob(os.path.join(RAW_PDF_DIR, "*.pdf"))
    
    if not pdf_queue:
        logger.warning("Tidak ditemukan file PDF di folder data/raw_pdf/!")
    else:
        logger.info(f"Ditemukan {len(pdf_queue)} file PDF siap diproses.")
        for pdf in pdf_queue:
            extractor.process_pdf(pdf)
            
    logger.info("Integrated Extraction Process Finished Successfully.")
