import logging
from src.agent.qdrant_client_tools import MultimodalQdrantRetriever

logger = logging.getLogger(__name__)

# Memanggil global dual retriever engine
multimodal_engine = MultimodalQdrantRetriever()

def search_academic_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    USE THIS TOOL to search the academic database for questions about scientific theories, 
    model architectures, tokenizer benchmarks, performance metrics (Accuracy, Fertility, F1-Score), 
    or textual analysis. Returns unified pages containing text and Markdown tables.
    """
    logger.info(f"TOOL CALL: search_academic_knowledge_base | Query: '{query}'")
    return multimodal_engine.search_text_and_tables(query=query, limit=top_k)

def search_academic_diagrams_and_plots(query: str, top_k: int = 2) -> str:
    """
    USE THIS TOOL ONLY WHEN the user explicitly asks to SEE, VIEW, or DISPLAY visual items 
    such as model architecture diagrams, flowcharts, training plots, visual schemas, or figures. 
    Returns the local file paths of the matching images.
    """
    logger.info(f"TOOL CALL: search_academic_diagrams_and_plots | Query: '{query}'")
    return multimodal_engine.search_images_via_clip(query=query, limit=top_k)

# Daftarkan kedua alat emas Anda!
AVAILABLE_TOOLS = [
    search_academic_knowledge_base,
    search_academic_diagrams_and_plots
]
