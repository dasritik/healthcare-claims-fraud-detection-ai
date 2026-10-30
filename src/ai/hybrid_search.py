
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "healthcare_claims"
COLLECTION_NAME = "healthcare_claims_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_CANDIDATES = 10
TOP_K = 5

# Reciprocal Rank Fusion parameters
RRF_K = 60


# ---------------------------------------------------------
# Text normalization and keyword extraction
# ---------------------------------------------------------

def tokenize(text):
    """Normalize text while retaining useful technical identifiers."""
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def keyword_score(question, document):
    """
    Calculate lexical overlap between the question and document.

    This is a simple baseline, not a calibrated relevance probability.
    """
    question_terms = tokenize(question)
    document_terms = tokenize(document)

    if not question_terms or not document_terms:
        return 0.0

    matched_terms = question_terms.intersection(document_terms)

    return len(matched_terms) / len(question_terms)


# ---------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------

def reciprocal_rank_fusion(semantic_results, lexical_results):
    """
    Combine semantic and lexical rankings.

    RRF score = sum(1 / (k + rank)) across ranking lists.
    Higher scores indicate stronger combined ranking.
    """
    fused_scores = {}

    for rank, item in enumerate(semantic_results, start=1):
        chunk_id = item["id"]
        fused_scores.setdefault(chunk_id, 0.0)
        fused_scores[chunk_id] += 1.0 / (RRF_K + rank)

    for rank, item in enumerate(lexical_results, start=1):
        chunk_id = item["id"]
        fused_scores.setdefault(chunk_id, 0.0)
        fused_scores[chunk_id] += 1.0 / (RRF_K + rank)

    return fused_scores


# ---------------------------------------------------------
# Load existing vector collection
# ---------------------------------------------------------

def load_collection():
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_PATH}"
        )

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Indexed chunks: {collection.count()}")

    return collection


# ---------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------

def hybrid_search(
    question,
    collection,
    embedding_model,
    top_k=TOP_K,
):
    count = collection.count()

    if count == 0:
        return []

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    # Retrieve the indexed corpus. This is practical for the current
    # small POC corpus; use a candidate-retrieval stage for large corpora.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    corpus = []

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        metadata = metadata or {}

        source = (
            metadata.get("source_filename")
            or metadata.get("source_file")
            or metadata.get("filename")
            or metadata.get("source")
            or "Unknown source"
        )

        page = (
            metadata.get("page_number")
            or metadata.get("page")
            or metadata.get("page_num")
            or "Unknown"
        )

        corpus.append({
            "id": chunk_id,
            "text": document or "",
            "source": source,
            "page": page,
            "distance": float(distance),
            "keyword_score": keyword_score(question, document or ""),
        })

    # Semantic ranking: Chroma's returned order is already distance-ranked.
    semantic_results = sorted(
        corpus,
        key=lambda item: item["distance"],
    )[:SEMANTIC_CANDIDATES]

    # Lexical ranking: prioritize direct term overlap.
    lexical_results = sorted(
        corpus,
        key=lambda item: item["keyword_score"],
        reverse=True,
    )[:SEMANTIC_CANDIDATES]

    fused_scores = reciprocal_rank_fusion(
        semantic_results,
        lexical_results,
    )

    semantic_rank = {
        item["id"]: rank
        for rank, item in enumerate(semantic_results, start=1)
    }

    lexical_rank = {
        item["id"]: rank
        for rank, item in enumerate(lexical_results, start=1)
    }

    for item in corpus:
        item["hybrid_score"] = fused_scores.get(item["id"], 0.0)
        item["semantic_rank"] = semantic_rank.get(item["id"])
        item["keyword_rank"] = lexical_rank.get(item["id"])

    ranked = sorted(
        corpus,
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return ranked[:top_k]


# ---------------------------------------------------------
# Interactive search
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("Healthcare Claims AI - Hybrid Document Search")
    print("=" * 70)

    collection = load_collection()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nAsk a question. Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Hybrid search closed.")
            break

        if not question:
            continue

        try:
            results = hybrid_search(
                question,
                collection,
                embedding_model,
            )

            print("\n" + "=" * 70)
            print("HYBRID SEARCH RESULTS")
            print("=" * 70)

            if not results:
                print("No document chunks found.")
                continue

            for rank, item in enumerate(results, start=1):
                print(
                    f"\nRank: {rank} | "
                    f"Hybrid score: {item['hybrid_score']:.5f}"
                )
                print(
                    f"Semantic rank: {item['semantic_rank']} | "
                    f"Keyword rank: {item['keyword_rank']}"
                )
                print(
                    f"Distance: {item['distance']:.4f} | "
                    f"Keyword overlap: {item['keyword_score']:.3f}"
                )
                print(f"Source: {item['source']}")
                print(f"Page: {item['page']}")
                print(f"Chunk ID: {item['id']}")
                print("\nRetrieved text:")
                print(item["text"])

        except Exception as exc:
            print(
                f"\nSearch failed: {type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()

