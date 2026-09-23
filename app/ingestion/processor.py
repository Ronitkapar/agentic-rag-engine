# to manage the parsers and processors for different data sources then storing it into Vector Database

import os
import sys
import uuid
import json
from datetime import datetime, timezone

import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import active_embedding_info, embed_texts
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

# Records which embedding backend built the current index. Dimensions alone
# cannot detect a provider switch (aicredits and google both emit 3072-dim
# vectors) but the two vector spaces are NOT interchangeable.
INDEX_META_PATH = os.path.join(PROCESSED_DATA_DIR, ".embedding_index_meta.json")


def _collection_dim(collection_name: str):
    """Vector size of an existing collection, or None when it cannot be read."""
    try:
        vectors = qdrant_client.get_collection(collection_name).config.params.vectors
        return getattr(vectors, "size", None)
    except Exception as e:
        logfire.warning(f"Could not read config for collection '{collection_name}': {e}")
        return None


def _read_index_fingerprint():
    if not os.path.exists(INDEX_META_PATH):
        return None
    try:
        with open(INDEX_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logfire.warning(f"Could not read {INDEX_META_PATH}: {e}")
        return None


def _check_index_fingerprint(info: dict):
    """Warn when the existing index was built with a different embedding backend."""
    previous = _read_index_fingerprint()
    if not previous:
        return
    previous_key = (previous.get("provider"), previous.get("model"), previous.get("dim"))
    current_key = (info["provider"], info["model"], info["dim"])
    if previous_key != current_key:
        logfire.warning(
            f"⚠️ Existing index was built with {previous_key[0]}/{previous_key[1]} "
            f"({previous_key[2]}-dim) but the active backend is {current_key[0]}/"
            f"{current_key[1]} ({current_key[2]}-dim). Vector spaces are not "
            f"interchangeable — re-run ingestion with --wipe."
        )


def _save_index_fingerprint(info: dict):
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "collection": settings.QDRANT_COLLECTION,
                    "provider": info["provider"],
                    "model": info["model"],
                    "dim": info["dim"],
                    "fallback": info["fallback"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logfire.warning(f"Could not write {INDEX_META_PATH}: {e}")


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Resolve the active embedding backend once — the dimension is probed
        # from the live provider, not hard-coded.
        info = active_embedding_info()
        logfire.info(
            f"Embedding backend: {info['provider']}/{info['model']} ({info['dim']}-dim)"
            + (" [LOCAL FALLBACK]" if info["fallback"] else "")
        )
        if info["probe_error"]:
            logfire.warning(f"Embedding probe error: {info['probe_error']}")
        _check_index_fingerprint(info)

        # Recreate collection if missing; refuse to write into a mismatched one.
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=info["dim"],
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({info['dim']}-dim, Cosine)."
            )
        else:
            existing_dim = _collection_dim(settings.QDRANT_COLLECTION)
            if existing_dim is not None and existing_dim != info["dim"]:
                raise RuntimeError(
                    f"Collection '{settings.QDRANT_COLLECTION}' is {existing_dim}-dim but the "
                    f"active embedding backend ({info['provider']}/{info['model']}) produces "
                    f"{info['dim']}-dim vectors. Qdrant vector size is immutable — recreate and "
                    f"re-index with:\n"
                    f"    python -m app.ingestion.processor {base_dir} --wipe"
                )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

        # Record which backend built the index so provider swaps are detectable.
        _save_index_fingerprint(info)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    logfire.info("Ingestion job completed.")

