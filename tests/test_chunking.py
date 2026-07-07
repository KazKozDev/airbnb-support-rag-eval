from src.ingest import chunk_fixed

def test_chunk_overlap_and_ids():
    pages = [{"page": 1, "text": " ".join(f"w{i}" for i in range(1000))}]
    chunks = chunk_fixed(pages, size=100, overlap=20)
    assert chunks[0]["id"] == "c0000"
    assert all(c["page"] == 1 for c in chunks)
    # overlap: last 20 words of chunk 0 == first 20 words of chunk 1
    w0 = chunks[0]["text"].split()
    w1 = chunks[1]["text"].split()
    assert w0[-20:] == w1[:20]

def test_tiny_tail_merged():
    pages = [{"page": 1, "text": " ".join(f"w{i}" for i in range(105))}]
    chunks = chunk_fixed(pages, size=100, overlap=0)
    assert len(chunks) == 1  # 5-word tail merged into previous
