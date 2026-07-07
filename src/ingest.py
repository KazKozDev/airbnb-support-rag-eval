"""PDF -> pages -> chunks (jsonl).

v0: fixed-size word-window chunking with overlap.
v2 idea: split on headings/structure before windowing (see chunk_structure_aware).
"""
import argparse
import json
import re

from pypdf import PdfReader

from src import config


def load_pdf(path: str) -> list[dict]:
    """Return list of {"page": int, "text": str} (1-indexed pages)."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_fixed(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Sliding word window across each page. Keeps page number for citations."""
    chunks = []
    idx = 0
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            piece = words[start : start + size]
            if len(piece) < 20 and chunks:  # merge tiny tail into previous chunk
                chunks[-1]["text"] += " " + " ".join(piece)
                break
            chunks.append({
                "id": f"c{idx:04d}",
                "page": p["page"],
                "text": " ".join(piece),
            })
            idx += 1
            start += size - overlap
    return chunks


# --- v4: boilerplate cleaning ---------------------------------------------
# The Help Center repeats site chrome in almost every chunk (the brand line and
# breadcrumb tags), which BM25 and embeddings latch onto as spurious signal.
# Cleaning is applied AFTER windowing, so chunk ids stay identical to the
# uncleaned run and golden-dataset relevant_chunk_ids remain valid — the recall
# delta is attributable to the text cleaning alone.
BOILERPLATE_PATTERNS = [
    r"Airbnb:\s*Vacation Rentals,\s*Cabins,\s*Beach Houses,\s*Unique Homes\s*&\s*Experiences",
    r"\b(How-to|Rules|Community policy|Guide)\s+(Home host|Experience host|Host|Guest)\b",
    r"\bRelated articles\b",
]
_BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS))


def clean_boilerplate(text: str) -> str:
    text = _BOILERPLATE_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


HEADING_RE = re.compile(r"^(\d+(\.\d+)*\s+.{3,80})$", re.M)


def chunk_structure_aware(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """v2: split on numbered headings first, then window inside sections."""
    sections = []
    for p in pages:
        parts = HEADING_RE.split(p["text"])
        # crude: fall back to whole page if no headings found
        texts = [t for t in parts if t and t.strip()] or [p["text"]]
        for t in texts:
            sections.append({"page": p["page"], "text": t.strip()})
    return chunk_fixed(sections, size, overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to source PDF")
    ap.add_argument("--strategy", choices=["fixed", "structure"], default="fixed")
    ap.add_argument("--clean", action="store_true",
                    help="v4: strip repeated site boilerplate from chunk text (ids unchanged)")
    args = ap.parse_args()

    pages = load_pdf(args.pdf)
    fn = chunk_fixed if args.strategy == "fixed" else chunk_structure_aware
    chunks = fn(pages, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    if args.clean:
        for c in chunks:
            c["text"] = clean_boilerplate(c["text"])

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(pages)} pages -> {len(chunks)} chunks -> {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
