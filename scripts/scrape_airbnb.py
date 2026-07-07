"""Scrape Airbnb Help Center articles into a single PDF for the RAG pipeline.

Airbnb's Help Center is a set of web pages, not one downloadable PDF. This script
fetches a list of article URLs, extracts the readable text, and writes one
`data/airbnb_help.pdf` that `src.ingest` can consume.

Usage:
    # 1. Put one article URL per line in scripts/airbnb_urls.txt
    #    (a starter list is provided below; add the articles you care about).
    python -m scripts.scrape_airbnb --urls scripts/airbnb_urls.txt --out data/airbnb_help.pdf

Notes / honest limitations:
- Airbnb serves content behind bot protection. Plain requests may return a
  near-empty page. If an article comes back with very little text, the script
  WARNS and skips it — open that URL in a browser, "Save as PDF", and merge it in
  manually, or paste the text into a .txt and feed that instead.
- This is a scraper for a personal eval project, not a crawler: it hits only the
  URLs you list, with a polite delay. Do not point it at the whole site.
"""
import argparse
import time

import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Starter list — edit scripts/airbnb_urls.txt to control exactly what gets indexed.
DEFAULT_URLS = [
    "https://www.airbnb.com/help/article/2868",   # cancellation policies for guests
    "https://www.airbnb.com/help/article/3218",   # AirCover for guests
    "https://www.airbnb.com/help/article/1320",   # getting a refund
    "https://www.airbnb.com/help/article/2701",   # check-in problems
    "https://www.airbnb.com/help/article/2503",   # change a reservation
]

MIN_CHARS = 400  # below this we assume the page was blocked / empty


def fetch(url: str) -> tuple[str, str]:
    """Return (title, body_text). Empty body means the fetch was not usable."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else url
    # main article text: prefer <main>/<article>, fall back to visible paragraphs
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.select("script, style, nav, header, footer"):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in root.find_all(["h1", "h2", "h3", "p", "li"])]
    body = "\n".join(t for t in paras if t)
    return title, body


def build_pdf(articles: list[tuple[str, str]], out: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out, pagesize=letter, title="Airbnb Help Center")
    flow = []
    for title, body in articles:
        flow.append(Paragraph(_esc(title), styles["Heading1"]))
        for line in body.split("\n"):
            flow.append(Paragraph(_esc(line), styles["BodyText"]))
        flow.append(Spacer(1, 24))
    doc.build(flow)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", help="text file with one article URL per line")
    ap.add_argument("--out", default="data/airbnb_help.pdf")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()

    if args.urls:
        urls = []
        with open(args.urls, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                urls.append(ln.split()[0])  # drop any inline "# comment" after the URL
    else:
        urls = DEFAULT_URLS

    articles = []
    for url in urls:
        try:
            title, body = fetch(url)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {url} -> {e}")
            continue
        if len(body) < MIN_CHARS:
            print(f"[SKIP] {url} -> only {len(body)} chars (likely blocked). "
                  f"Save this page as PDF manually and merge it in.")
            continue
        print(f"[OK]   {url} -> {len(body)} chars")
        articles.append((title, body))
        time.sleep(args.delay)

    if not articles:
        print("No usable articles. See the note about bot protection at the top of this file.")
        return
    build_pdf(articles, args.out)
    print(f"\nWrote {len(articles)} articles -> {args.out}\n"
          f"Next: python -m src.ingest {args.out} --strategy fixed && python -m src.index")


if __name__ == "__main__":
    main()
