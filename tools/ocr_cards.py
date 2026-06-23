"""OCR every card image out of data/Card_ID List_EN.pdf -> data/card_ocr.csv (cardId, name, ocr_text).
Page mapping: page = 39 + cardId (verified: p40=card1, 743=Alakazam, 121=Dragapult ex). The card
art garbles but the text blocks (name/HP/attacks/effects/ability) OCR cleanly. Authoritative card
text for the full pool — fills the ~541 cards EN_Card_Data.csv lacks, for generative deck-building.

  python tools/ocr_cards.py [workers]
"""
import csv, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = f"{ROOT}/data/Card_ID List_EN.pdf"
OUT = f"{ROOT}/data/card_ocr.csv"
NAMES = {int(r["cardId"]): r["name"] for r in csv.DictReader(open(f"{ROOT}/autoresearch/cards_full.csv"))}


def clean(text):
    keep = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # drop art-noise lines: mostly non-alpha or too few real words
        alpha = sum(c.isalpha() for c in ln)
        if alpha < 4 or alpha < len(ln) * 0.45:
            continue
        keep.append(re.sub(r"\s+", " ", ln))
    return " \n".join(keep)


def ocr_one(cid):
    page = 39 + cid
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdfimages", "-f", str(page), "-l", str(page), "-j", PDF, f"{td}/c"],
                       capture_output=True, timeout=60)
        jpg = f"{td}/c-000.jpg"
        if not os.path.exists(jpg):
            return cid, NAMES.get(cid, "?"), ""
        r = subprocess.run(["tesseract", jpg, "-", "--psm", "6"],
                           capture_output=True, text=True, timeout=60, cwd=td)
        return cid, NAMES.get(cid, "?"), clean(r.stdout)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cids = sorted(NAMES)
    print(f"OCR {len(cids)} cards, {workers} workers -> {OUT}", flush=True)
    rows = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cid, name, text in pool.map(ocr_one, cids):
            rows[cid] = (name, text)
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(cids)}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cardId", "name", "ocr_text"])
        for cid in cids:
            w.writerow([cid, rows[cid][0], rows[cid][1]])
    empty = sum(1 for cid in cids if not rows[cid][1])
    print(f"done. {len(cids)} cards, {empty} empty (image-only/energy).", flush=True)


if __name__ == "__main__":
    main()
