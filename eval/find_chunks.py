"""Helper for annotation: shortlist candidate chunks for a question."""
import sys
from src.retrieval import Retriever

if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    r = Retriever()
    for c in r.retrieve(query, k=8, mode="hybrid"):
        print(f"--- {c['id']} (page {c['page']}) ---")
        print(c["text"][:400], "\n")
