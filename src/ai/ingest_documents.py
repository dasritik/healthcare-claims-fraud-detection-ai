
from pathlib import Path
import re
import json
import hashlib
import logging
from datetime import datetime

import pandas as pd
import pymupdf
import pdfplumber


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "healthcare_claims_ai_generated"
    / "documents"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSET_DIR = PROJECT_ROOT / "extracted_assets" / "figures"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".csv",
    ".docx",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. TEXT CLEANING AND CHUNKING
# ============================================================

def clean_text(text):
    """Normalize extracted text without removing useful punctuation."""

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(text, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring paragraph boundaries."""

    text = clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind("\n", start, end)

            if boundary > start + int(chunk_size * 0.6):
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def make_document_id(path):
    """Create a stable identifier from a relative document path."""

    relative_path = str(path.relative_to(DOCUMENTS_DIR)).replace("\\", "/")

    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# 3. REFERENCE DETECTION
# ============================================================

REFERENCE_PATTERNS = {
    "rule_id": r"\b(?:RULE|RISK_RULE|RULE_ID)[\s_-]*[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b",
    "section": r"\bSection\s+\d+(?:\.\d+)*\b",
    "table": r"\bTable\s+\d+(?:\.\d+)*\b",
    "figure": r"\bFigure\s+\d+(?:\.\d+)*\b",
    "policy_reference": r"\b(?:Policy|Procedure|Control)\s+(?:ID[\s:#-]*)?[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b",
}


def extract_references(text, document_id, filename, page_number):
    """Find reference-like identifiers in extracted text."""

    results = []

    for reference_type, pattern in REFERENCE_PATTERNS.items():
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)

        for match in matches:
            results.append({
                "document_id": document_id,
                "source_file": filename,
                "page_number": page_number,
                "reference_type": reference_type,
                "reference_value": match.group(0).strip(),
                "context": text[
                    max(0, match.start() - 100):
                    min(len(text), match.end() + 100)
                ].replace("\n", " "),
            })

    return results


# ============================================================
# 4. PDF TEXT, TABLE AND FIGURE EXTRACTION
# ============================================================

def extract_pdf(path, document_id):
    """Extract page text, tables, embedded images and visual page assets."""

    chunks_out = []
    tables_out = []
    figures_out = []
    references_out = []
    page_count = 0
    pages_with_text = 0
    extraction_warnings = []

    relative_path = str(path.relative_to(DOCUMENTS_DIR)).replace("\\", "/")

    # --------------------------------------------------------
    # 4.1 Extract text and embedded images using PyMuPDF
    # --------------------------------------------------------

    with pymupdf.open(path) as pdf:
        page_count = len(pdf)

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_text = clean_text(page.get_text("text"))

            if page_text:
                pages_with_text += 1

                chunks = split_into_chunks(page_text)

                for chunk_index, chunk in enumerate(chunks):
                    chunks_out.append({
                        "document_id": document_id,
                        "source_file": path.name,
                        "source_path": relative_path,
                        "file_type": "pdf",
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": (
                            f"{document_id}_p{page_number}_c{chunk_index}"
                        ),
                        "text": chunk,
                        "character_count": len(chunk),
                    })

                references_out.extend(
                    extract_references(
                        page_text,
                        document_id,
                        path.name,
                        page_number,
                    )
                )

            # Extract embedded raster images.
            image_list = page.get_images(full=True)

            for image_index, image_info in enumerate(image_list):
                xref = image_info[0]

                try:
                    image_data = pdf.extract_image(xref)
                    image_bytes = image_data["image"]
                    image_extension = image_data.get("ext", "bin")

                    asset_name = (
                        f"{document_id}_p{page_number}"
                        f"_img{image_index}.{image_extension}"
                    )

                    asset_path = ASSET_DIR / asset_name
                    asset_path.write_bytes(image_bytes)

                    figures_out.append({
                        "document_id": document_id,
                        "source_file": path.name,
                        "source_path": relative_path,
                        "page_number": page_number,
                        "figure_index": image_index,
                        "asset_type": "embedded_image",
                        "asset_path": str(
                            asset_path.relative_to(PROJECT_ROOT)
                        ).replace("\\", "/"),
                        "image_width": image_data.get("width"),
                        "image_height": image_data.get("height"),
                        "caption_or_context": page_text[:500],
                    })

                except Exception as exc:
                    extraction_warnings.append(
                        f"Page {page_number}, image {image_index}: {exc}"
                    )

            # Render pages that contain embedded images or vector drawings.
            # This can preserve charts/diagrams that are not separate images.
            has_drawings = bool(page.get_drawings())

            if image_list or has_drawings:
                try:
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(1.5, 1.5),
                        alpha=False,
                    )

                    render_name = (
                        f"{document_id}_p{page_number}_render.png"
                    )
                    render_path = ASSET_DIR / render_name
                    pixmap.save(str(render_path))

                    figures_out.append({
                        "document_id": document_id,
                        "source_file": path.name,
                        "source_path": relative_path,
                        "page_number": page_number,
                        "figure_index": None,
                        "asset_type": "page_render",
                        "asset_path": str(
                            render_path.relative_to(PROJECT_ROOT)
                        ).replace("\\", "/"),
                        "image_width": pixmap.width,
                        "image_height": pixmap.height,
                        "caption_or_context": page_text[:500],
                    })

                except Exception as exc:
                    extraction_warnings.append(
                        f"Page {page_number}, page render: {exc}"
                    )

    # --------------------------------------------------------
    # 4.2 Extract tables using pdfplumber
    # --------------------------------------------------------

    try:
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1

                try:
                    page_tables = page.extract_tables() or []

                    for table_index, table in enumerate(page_tables):
                        if not table:
                            continue

                        normalized_table = []

                        for row in table:
                            normalized_table.append([
                                clean_text(str(cell)) if cell is not None else ""
                                for cell in row
                            ])

                        # Ignore tables that are entirely empty.
                        if not any(
                            cell
                            for row in normalized_table
                            for cell in row
                        ):
                            continue

                        tables_out.append({
                            "document_id": document_id,
                            "source_file": path.name,
                            "source_path": relative_path,
                            "page_number": page_number,
                            "table_index": table_index,
                            "row_count": len(normalized_table),
                            "column_count": max(
                                (len(row) for row in normalized_table),
                                default=0,
                            ),
                            "table_json": json.dumps(
                                normalized_table,
                                ensure_ascii=False,
                            ),
                        })

                except Exception as exc:
                    extraction_warnings.append(
                        f"Page {page_number}, table extraction: {exc}"
                    )

    except Exception as exc:
        extraction_warnings.append(f"pdfplumber failed: {exc}")

    return {
        "chunks": chunks_out,
        "tables": tables_out,
        "figures": figures_out,
        "references": references_out,
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "warnings": extraction_warnings,
    }


# ============================================================
# 5. NON-PDF DOCUMENT EXTRACTION
# ============================================================

def extract_non_pdf(path, document_id):
    """Extract text from TXT, Markdown, CSV and DOCX files."""

    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".csv":
        dataframe = pd.read_csv(path, dtype=str).fillna("")
        text = dataframe.to_string(index=False)

    elif suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "DOCX support requires python-docx. Install with: "
                "python -m pip install python-docx"
            ) from exc

        document = Document(path)
        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

    else:
        raise ValueError(f"Unsupported extension: {suffix}")

    relative_path = str(path.relative_to(DOCUMENTS_DIR)).replace("\\", "/")
    text = clean_text(text)

    chunks_out = []

    for chunk_index, chunk in enumerate(split_into_chunks(text)):
        chunks_out.append({
            "document_id": document_id,
            "source_file": path.name,
            "source_path": relative_path,
            "file_type": suffix.lstrip("."),
            "page_number": None,
            "chunk_index": chunk_index,
            "chunk_id": f"{document_id}_c{chunk_index}",
            "text": chunk,
            "character_count": len(chunk),
        })

    references_out = extract_references(
        text,
        document_id,
        path.name,
        None,
    )

    return {
        "chunks": chunks_out,
        "tables": [],
        "figures": [],
        "references": references_out,
        "page_count": None,
        "pages_with_text": None,
        "warnings": [],
    }


# ============================================================
# 6. MAIN INGESTION PIPELINE
# ============================================================

def main():
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    all_chunks = []
    all_tables = []
    all_figures = []
    all_references = []
    inventory = []

    files = sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Documents folder: %s", DOCUMENTS_DIR)
    logger.info("Found %s supported documents.", len(files))

    for path in files:
        document_id = make_document_id(path)
        relative_path = str(
            path.relative_to(DOCUMENTS_DIR)
        ).replace("\\", "/")

        logger.info("Processing: %s", relative_path)

        started_at = datetime.now().isoformat(timespec="seconds")
        status = "SUCCESS"
        error_message = ""

        try:
            if path.suffix.lower() == ".pdf":
                result = extract_pdf(path, document_id)
            else:
                result = extract_non_pdf(path, document_id)

            all_chunks.extend(result["chunks"])
            all_tables.extend(result["tables"])
            all_figures.extend(result["figures"])
            all_references.extend(result["references"])

            if not result["chunks"]:
                status = "NO_TEXT_EXTRACTED"

            if result["warnings"]:
                error_message = " | ".join(result["warnings"])

            inventory.append({
                "document_id": document_id,
                "source_file": path.name,
                "source_path": relative_path,
                "file_type": path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size,
                "page_count": result["page_count"],
                "pages_with_text": result["pages_with_text"],
                "chunk_count": len(result["chunks"]),
                "table_count": len(result["tables"]),
                "figure_asset_count": len(result["figures"]),
                "reference_count": len(result["references"]),
                "status": status,
                "warning_or_error": error_message,
                "processed_at": started_at,
            })

        except Exception as exc:
            logger.exception("Failed to process %s", relative_path)

            inventory.append({
                "document_id": document_id,
                "source_file": path.name,
                "source_path": relative_path,
                "file_type": path.suffix.lower().lstrip("."),
                "file_size_bytes": path.stat().st_size,
                "page_count": None,
                "pages_with_text": None,
                "chunk_count": 0,
                "table_count": 0,
                "figure_asset_count": 0,
                "reference_count": 0,
                "status": "FAILED",
                "warning_or_error": str(exc),
                "processed_at": started_at,
            })

    # --------------------------------------------------------
    # 6.1 Save CSV outputs
    # --------------------------------------------------------

    output_files = {
        "document_chunks.csv": all_chunks,
        "document_tables.csv": all_tables,
        "document_figures.csv": all_figures,
        "document_inventory.csv": inventory,
        "document_references.csv": all_references,
    }

    for filename, records in output_files.items():
        output_path = OUTPUT_DIR / filename
        dataframe = pd.DataFrame(records)
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("Saved %s (%s records)", output_path, len(dataframe))

    # --------------------------------------------------------
    # 6.2 Print summary
    # --------------------------------------------------------

    inventory_df = pd.DataFrame(inventory)

    print("\n" + "=" * 65)
    print("DOCUMENT INGESTION SUMMARY")
    print("=" * 65)
    print(f"Documents discovered: {len(files)}")
    print(f"Chunks extracted:     {len(all_chunks)}")
    print(f"Tables extracted:     {len(all_tables)}")
    print(f"Figure assets saved:  {len(all_figures)}")
    print(f"References found:     {len(all_references)}")

    if not inventory_df.empty:
        print("\nDocument status:")
        print(inventory_df["status"].value_counts().to_string())

        print("\nPer-document results:")
        columns = [
            "source_file",
            "page_count",
            "chunk_count",
            "table_count",
            "figure_asset_count",
            "reference_count",
            "status",
        ]
        print(inventory_df[columns].to_string(index=False))

        problem_rows = inventory_df[
            inventory_df["status"] != "SUCCESS"
        ]

        if not problem_rows.empty:
            print("\nDocuments requiring attention:")
            print(
                problem_rows[
                    ["source_file", "status", "warning_or_error"]
                ].to_string(index=False)
            )

    print("\nOutput folder:", OUTPUT_DIR)
    print("Figure assets:", ASSET_DIR)
    print("\nIngestion completed.")


if __name__ == "__main__":
    main()
