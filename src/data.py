"""Corpus readers; downloads and caches stay inside this project."""

import random
from pathlib import Path

import nltk

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data" / "nltk_data"
# Fallback to parent repository data directory if not found locally
if not DATA_DIR.exists():
    PARENT_DATA = PACKAGE_ROOT.parent / "data" / "nltk_data"
    if PARENT_DATA.exists():
        DATA_DIR = PARENT_DATA


def load_corpus(name="pride"):
    readers = {
        "pride": "gutenberg",
        "pride_and_prejudice": "gutenberg",
        "brown": "brown",
        "reuters": "reuters",
    }
    if name not in readers:
        raise ValueError(f"Corpus harus salah satu {tuple(readers)}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(DATA_DIR))
    resource = readers[name]
    try:
        try:
            nltk.data.find(f"corpora/{resource}.zip")
        except LookupError:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        if not nltk.download(resource, download_dir=str(DATA_DIR), quiet=True):
            raise RuntimeError(f"Download {resource} gagal; periksa koneksi")
    if name in ("pride", "pride_and_prejudice", "reuters"):
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            if not nltk.download("punkt_tab", download_dir=str(DATA_DIR), quiet=True):
                raise RuntimeError("Download tokenizer punkt_tab gagal")
    from nltk.corpus import brown, gutenberg, reuters

    if name in ("pride", "pride_and_prejudice"):
        pride_path = DATA_DIR / "corpora" / "gutenberg" / "austen-pride.txt"
        if not pride_path.exists():
            import urllib.request
            url = "https://www.gutenberg.org/files/1342/1342-0.txt"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                text = resp.read().decode("utf-8-sig", errors="ignore")
            text = text.replace("\u2018", "'").replace("\u2019", "'")
            text = text.replace("\u201c", '"').replace("\u201d", '"')
            text = text.replace("\u2014", " -- ").replace("\u2013", " - ")
            pride_path.parent.mkdir(parents=True, exist_ok=True)
            pride_path.write_bytes(text.encode("latin-1", errors="ignore"))
        sentences = gutenberg.sents("austen-pride.txt")
    elif name == "reuters":
        sentences = reuters.sents()
    else:
        sentences = brown.sents()

    return [list(sentence) for sentence in sentences]


def set_seed(seed=42):
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
