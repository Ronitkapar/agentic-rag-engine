import logfire
from qdrant_client import QdrantClient
from app.config import settings
from app.services.retrieval.embedding import embed_query

# Lazy initialization - the client is built on first use so that importing
# app.main never depends on Qdrant being configured or reachable. An unset
# QDRANT_CLUSTER_ENDPOINT used to raise at import time, which turned a missing
# env var (or a brief Qdrant outage at boot) into a container crashloop
# instead of a failed query. Mirrors _get_ranker() in ranking_service.py.
_client = None


def _get_client() -> QdrantClient:
    """Initialize the Qdrant client lazily, once per process."""
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _client


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = _get_client().query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "score": res.score
            })
        
        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
 