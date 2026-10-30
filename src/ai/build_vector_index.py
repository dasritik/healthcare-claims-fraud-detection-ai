
from pathlib import Path
import logging

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_CSV = PROJECT_ROOT / "outputs" / "document_chunks.csv"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_store" / "healthcare_claims"
COLLECTION_NAME = "healthcare_claims_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. LOAD AND VALIDATE CHUNKS
# ============================================================

def load_chunks():
    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(
            f"Document chunks CSV not found: {CHUNKS_CSV}"
        )

    df = pd.read_csv(CHUNKS_CSV, dtype=str).fillna("")

    required_columns = [
        "chunk_id",
        "document_id",
        "source_file",
        "source_path",
        "text",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in document_chunks.csv: "
            f"{missing_columns}"
        )

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    # Keep one record per chunk ID to avoid duplicate vector IDs.
    df = df.drop_duplicates(subset=["chunk_id"])

    if df.empty:
        raise ValueError("No non-empty document chunks found.")

    logger.info("Loaded %s text chunks.", len(df))

    return df


# ============================================================
# 3. CREATE DOCUMENT METADATA
# ============================================================

def build_metadata(row):
    metadata = {
        "document_id": str(row["document_id"]),
        "source_file": str(row["source_file"]),
        "source_path": str(row["source_path"]),
    }

    # ChromaDB metadata values must be simple supported types.
    for column in ["file_type", "page_number", "chunk_index"]:
        if column not in row.index:
            continue

        value = str(row[column]).strip()

        if not value:
            continue

        if column in {"page_number", "chunk_index"}:
            try:
                metadata[column] = int(float(value))
            except (ValueError, TypeError):
                pass
        else:
            metadata[column] = value

    return metadata


# ============================================================
# 4. BUILD OR REBUILD VECTOR INDEX
# ============================================================

def main():
    df = load_chunks()

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Initializing persistent ChromaDB client.")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    # Rebuild this project's document collection on every run.
    # This prevents stale chunks from remaining after re-ingestion.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Deleted existing collection for a clean rebuild.")
    except Exception:
        logger.info("No existing collection found; creating a new one.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Healthcare insurance claims policy, rulebook, "
                "SOP and data dictionary document chunks"
            ),
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        embedding_function=None,
    )

    total_chunks = len(df)

    for start in range(0, total_chunks, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]

        texts = batch["text"].tolist()
        ids = batch["chunk_id"].tolist()
        metadatas = [
            build_metadata(row)
            for _, row in batch.iterrows()
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        logger.info(
            "Indexed %s / %s chunks.",
            min(start + len(batch), total_chunks),
            total_chunks,
        )

    # --------------------------------------------------------
    # 4.1 Validate the saved collection
    # --------------------------------------------------------

    stored_count = collection.count()

    if stored_count != total_chunks:
        raise RuntimeError(
            f"Index validation failed: expected {total_chunks} chunks, "
            f"but ChromaDB contains {stored_count}."
        )

    print("\n" + "=" * 60)
    print("VECTOR INDEX BUILD SUMMARY")
    print("=" * 60)
    print(f"Input CSV:       {CHUNKS_CSV}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Collection:      {COLLECTION_NAME}")
    print(f"Chunks indexed:  {stored_count}")
    print(f"Vector store:    {VECTOR_DB_DIR}")
    print("Validation:      PASSED")


if __name__ == "__main__":
    main()
