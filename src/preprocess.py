"""Sentence-level preprocessing, data splitting, and training-only vocabulary."""

import random
from collections import Counter

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"


def tokenize(text):
    from nltk.tokenize import wordpunct_tokenize

    return wordpunct_tokenize(text)


def preprocess_text(text_or_sentence):
    """Task 2: Implement basic text preprocessing.
    1. convert text to lowercase
    2. remove unnecessary characters (non-alphabetic)
    3. tokenize the text
    4. remove empty tokens
    """
    if isinstance(text_or_sentence, str):
        tokens = tokenize(text_or_sentence)
    else:
        tokens = text_or_sentence
    return [w.lower() for w in tokens if w.isalpha()]


def preprocess(sentences):
    processed = [preprocess_text(s) for s in sentences]
    return [s for s in processed if s], sum(not s for s in processed)


def split_sentences(sentences, train_ratio=0.8, seed=42):
    if not 0 < train_ratio < 1 or len(sentences) < 2:
        raise ValueError("Split memerlukan >=2 kalimat dan rasio antara 0 dan 1")
    ids = list(range(len(sentences)))
    random.Random(seed).shuffle(ids)
    cut = max(1, min(len(ids) - 1, int(len(ids) * train_ratio)))
    train_ids, test_ids = ids[:cut], ids[cut:]
    return (
        [sentences[i] for i in train_ids],
        [sentences[i] for i in test_ids],
        train_ids,
        test_ids,
    )


# Task 4 alias
split_corpus = split_sentences


def fit_vocabulary(train, min_count=2):
    if min_count < 1:
        raise ValueError("min_count harus >=1")
    counts = Counter(w for sentence in train for w in sentence)
    vocabulary = {w for w, count in counts.items() if count >= min_count}
    return vocabulary | {UNK, EOS}, counts


def replace_unknown(sentences, vocabulary):
    return [[w if w in vocabulary else UNK for w in sentence] for sentence in sentences]


def oov_rate(sentences, vocabulary):
    tokens = [w for sentence in sentences for w in sentence]
    return sum(w not in vocabulary for w in tokens) / len(tokens) if tokens else 0.0
