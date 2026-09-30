import time
import logfire
from flashrank import Ranker, RerankRequest

from app.config import settings

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-TinyBERT-L-2-v2) for ultra-fast reranking.

    max_length caps tokenisation per passage, which is what bounds the ONNX
    session's memory (see RERANK_MAX_LENGTH in config).
    """
    global _ranker
    if _ranker is None:
        logfire.info(
            f"🧠 Initializing FlashRank Model (TinyBERT) locally... "
            f"max_length={settings.RERANK_MAX_LENGTH}"
        )
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(
                cache_dir=settings.RERANK_CACHE_DIR,
                max_length=settings.RERANK_MAX_LENGTH,
            )
        except Exception:
            _ranker = Ranker(max_length=settings.RERANK_MAX_LENGTH)
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    if not settings.RERANK_ENABLED:
        # RERANK_ENABLED=false — keep Qdrant's vector order, never load the ONNX
        # session (saves its RAM/CPU entirely on the most constrained hosts).
        logfire.info("Reranking disabled (RERANK_ENABLED=false) — keeping vector order.")
        return documents[:top_n]

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        # Results are returned sorted by highest semantic score first
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        
        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]
