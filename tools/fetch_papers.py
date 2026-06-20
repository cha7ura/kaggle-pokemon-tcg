"""Fetch arxiv papers -> clean markdown under wiki/papers/. Dev tooling only; not shipped."""
import io
import os
import re
import sys
import urllib.request


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def clean_markdown(raw_text: str) -> str:
    text = raw_text.replace("\x0c", "\n")          # pdf page breaks -> newline
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces/tabs
    text = re.sub(r" *\n", "\n", text)             # trim trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)         # max one blank line
    return text.strip()


def _extract_pdf_text(pdf_bytes: bytes):
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return None


def fetch_paper(arxiv_id: str, out_dir: str = "wiki/papers", retrieved: str | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{arxiv_id}.md")
    url = arxiv_pdf_url(arxiv_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper-fetch)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        pdf_bytes = resp.read()
    text = _extract_pdf_text(pdf_bytes)
    if text and len(text.strip()) > 500:
        body = clean_markdown(text)
        header = f"# {arxiv_id}\n\nSource: {url}\nRetrieved: {retrieved or ''}\n\n---\n\n"
    else:
        body = ("(PDF text extraction unavailable — install `pypdf` or add a WebFetch summary here. "
                "This stub marks the paper as fetched-but-not-extracted.)")
        header = f"# {arxiv_id}\n\nSource: {url}\nRetrieved: {retrieved or ''}\nStatus: STUB\n\n---\n\n"
    with open(out_path, "w") as f:
        f.write(header + body + "\n")
    return out_path


if __name__ == "__main__":
    for aid in sys.argv[1:]:
        print(fetch_paper(aid))
