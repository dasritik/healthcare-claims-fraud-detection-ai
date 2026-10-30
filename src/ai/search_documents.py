
from pathlib import Path
import logging

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = PROJECT_ROOT / "vector_store" / "healthcare_claims"
COLLECTION_NAME = "healthcare_claims_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. LOAD VECTOR INDEX AND EMBEDDING MODEL
# ============================================================

def load_vector_collection():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_DB_DIR}. "
            "Run src/ai/build_vector_index.py first."
        )

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Build the vector index first."
        ) from exc

    if collection.count() == 0:
        raise RuntimeError(
            "The vector collection is empty. Rebuild the index."
        )

    logger.info(
        "Connected to collection '%s' with %s chunks.",
        COLLECTION_NAME,
        collection.count(),
    )

    return client, collection


def load_embedding_model():
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# 3. SEMANTIC SEARCH
# ============================================================

def search_documents(question, collection, model, top_k=DEFAULT_TOP_K):
    question = question.strip()

    if not question:
        raise ValueError("Please enter a non-empty question.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    available_chunks = collection.count()
    result_count = min(top_k, available_chunks)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}

        matches.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "source_file": metadata.get("source_file", "Unknown"),
            "source_path": metadata.get("source_path", "Unknown"),
            "page_number": metadata.get("page_number"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": float(distance),
            "text": document or "",
        })

    return matches


# ============================================================
# 4. DISPLAY RESULTS
# ============================================================

def display_results(question, matches):
    print("\n" + "=" * 80)
    print("SEMANTIC DOCUMENT SEARCH")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Results returned: {len(matches)}")

    if not matches:
        print("\nNo matching document chunks were returned.")
        return

    for match in matches:
        print("\n" + "-" * 80)
        print(
            f"Rank: {match['rank']} | "
            f"Distance: {match['distance']:.4f}"
        )
        print(f"Source: {match['source_file']}")
        print(f"Path: {match['source_path']}")

        if match["page_number"] is not None:
            print(f"Page: {match['page_number']}")

        print(f"Chunk ID: {match['chunk_id']}")
        print("\nRetrieved text:")
        print(match["text"])

    print("\n" + "=" * 80)
    print(
        "Note: Lower distance generally indicates a closer vector match. "
        "Review the retrieved text before treating it as supporting evidence."
    )


# ============================================================
# 5. INTERACTIVE SEARCH LOOP
# ============================================================

def main():
    _, collection = load_vector_collection()
    model = load_embedding_model()

    print("\nHealthcare Claims AI - Document Search")
    print("Ask questions about your indexed documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Document search closed.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            matches = search_documents(
                question=question,
                collection=collection,
                model=model,
                top_k=DEFAULT_TOP_K,
            )

            display_results(question, matches)

        except Exception as exc:
            logger.exception("Search failed: %s", exc)
            print(f"Search error: {exc}")


if __name__ == "__main__":
    main()
