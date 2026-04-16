"""Load business rules and documentation into ChromaDB for vector search."""

import re
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "vectordb"


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split markdown into chunks by heading sections."""
    chunks = []
    sections = re.split(r"\n(?=## )", text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract heading if present
        lines = section.split("\n", 1)
        heading = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else heading

        chunks.append({
            "id": f"{source}::{heading}".replace(" ", "_").lower(),
            "text": section,
            "metadata": {
                "source": source,
                "heading": heading,
            },
        })
    return chunks


def main() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection for clean load
    try:
        client.delete_collection("business_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="business_docs",
        metadata={"description": "Business rules, KPI definitions, and data dictionary"},
    )

    all_chunks = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        text = md_file.read_text()
        source = md_file.stem
        chunks = chunk_markdown(text, source)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    print(f"\nLoaded {len(all_chunks)} chunks into ChromaDB at {CHROMA_DIR}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
