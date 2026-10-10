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


DAILYDIALOG_URL = "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip"
DAILYDIALOG_DIR = PACKAGE_ROOT / "data" / "dailydialog"

# Contraction suffixes -> expanded words (our preprocessing keeps alphabetic tokens only,
# so "I ' m" would otherwise become the broken pair "i m").
_CONTRACTIONS = {"m": "am", "re": "are", "ve": "have", "ll": "will", "d": "would"}
_S_IS_HOSTS = {"it", "that", "he", "she", "what", "there", "here", "who", "where", "how", "let"}
_NT_SPECIAL = {"won": "will", "can": "can", "shan": "shall", "ain": "is"}


def _expand_contractions(tokens):
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        nxt2 = tokens[i + 2] if i + 2 < len(tokens) else None
        if nxt == "'" and nxt2 is not None:
            suf = nxt2.lower()
            if suf == "t" and tok.lower().endswith("n"):
                base = tok[:-1]
                out.append(_NT_SPECIAL.get(tok.lower(), base) or base)
                out.append("not")
                i += 3
                continue
            if suf in _CONTRACTIONS:
                out += [tok, _CONTRACTIONS[suf]]
                i += 3
                continue
            if suf == "s":
                out.append(tok)
                if tok.lower() == "let":
                    out.append("us")
                elif tok.lower() in _S_IS_HOSTS:
                    out.append("is")
                i += 3
                continue
        out.append(tok)
        i += 1
    return out


def load_daily_dialog():
    """Load DailyDialog (train split) as a list of tokenized sentences.

    Each dialogue line holds utterances separated by '__eou__'. Utterances are split
    further into sentences on terminal punctuation.
    """
    path = DAILYDIALOG_DIR / "dialogues_train.txt"
    if not path.exists():
        import io
        import urllib.request
        import zipfile

        req = urllib.request.Request(DAILYDIALOG_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            content = z.read("train/dialogues_train.txt")
        DAILYDIALOG_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    sentences = []
    for line in text.splitlines():
        for utt in line.split("__eou__"):
            tokens = _expand_contractions(utt.split())
            current = []
            for tok in tokens:
                current.append(tok)
                if tok in (".", "?", "!"):
                    sentences.append(current)
                    current = []
            if current:
                sentences.append(current)
    return [s for s in sentences if s]


def load_dialogue_pairs():
    """Load consecutive (turn_1, turn_2) prompt-response dialogue pairs from DailyDialog."""
    load_daily_dialog()
    path = DAILYDIALOG_DIR / "dialogues_train.txt"
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    pairs = []
    for line in text.splitlines():
        turns = [u.strip() for u in line.split("__eou__") if u.strip()]
        for i in range(len(turns) - 1):
            if turns[i] and turns[i + 1]:
                pairs.append((turns[i], turns[i + 1]))
    return pairs


def load_corpus(name="pride"):
    if name in ("dailydialog", "daily_dialog"):
        return load_daily_dialog()
    readers = {
        "pride": "gutenberg",
        "pride_and_prejudice": "gutenberg",
        "brown": "brown",
        "reuters": "reuters",
    }
    if name not in readers:
        raise ValueError(f"Corpus harus salah satu {tuple(readers) + ('dailydialog',)}")
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
