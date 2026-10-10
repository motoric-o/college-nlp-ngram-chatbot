"""Generate 5 sentences for Core N-Gram Models (Tasks 16 & 17 Deliverables).

Models covered:
  1. Bigram Model (Assignment Deliverable 3)
  2. Trigram Model (Assignment Deliverable 3)
  3. Unigram Model (Baseline)
  4. Laplace-Smoothed Bigram Model

Outputs:
  - Console display
  - Saved to 'generated_sentences.txt' in the same folder.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import load_corpus, set_seed
from src.preprocess import preprocess_text, split_corpus
from src.ngram import (
    BOS,
    EOS,
    add_sentence_boundaries,
    generate_ngrams,
    count_ngrams,
    generate_sentence,
    sample_with_temperature,
)


def main():
    set_seed(42)
    print("[*] Loading corpus 'pride'...")
    raw_sents = load_corpus("pride")
    clean_sents = [preprocess_text(s) for s in raw_sents]
    clean_sents = [s for s in clean_sents if s]
    bounded_sents = [add_sentence_boundaries(s) for s in clean_sents]
    train_sents, _, _, _ = split_corpus(bounded_sents, train_ratio=0.8, seed=42)

    unigrams = [u for s in train_sents for u in generate_ngrams(s, 1)]
    bigrams = [b for s in train_sents for b in generate_ngrams(s, 2)]
    trigrams = [t for s in train_sents for t in generate_ngrams(s, 3)]

    uni_counts = count_ngrams(unigrams)
    bi_counts = count_ngrams(bigrams)
    tri_counts = count_ngrams(trigrams)
    vocab = set(w for b in bi_counts for w in b if w not in (BOS, EOS))

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 75)
    log("  DELIVERABLE 3: FIVE GENERATED SENTENCES FOR N-GRAM MODELS")
    log("  Corpus: Jane Austen's 'Pride and Prejudice'")
    log("=" * 75 + "\n")

    # 1. Bigram Model (Required by PDF)
    log("---------------------------------------------------------------------------")
    log("1. BIGRAM MODEL (MLE Distribution - Assignment Deliverable 3)")
    log("---------------------------------------------------------------------------")
    set_seed(42)
    for i in range(1, 6):
        sent = generate_sentence(bi_counts, n=2, max_len=25, seed=42 + i * 11)
        log(f"  [{i}] {sent}")
    log()

    # 2. Trigram Model (Required by PDF)
    log("---------------------------------------------------------------------------")
    log("2. TRIGRAM MODEL (MLE Distribution - Assignment Deliverable 3)")
    log("---------------------------------------------------------------------------")
    set_seed(42)
    for i in range(1, 6):
        sent = generate_sentence(tri_counts, n=3, max_len=25, seed=100 + i * 17)
        log(f"  [{i}] {sent}")
    log()

    # 3. Unigram Model (Baseline Comparison)
    log("---------------------------------------------------------------------------")
    log("3. UNIGRAM MODEL (Word Independence Baseline)")
    log("---------------------------------------------------------------------------")
    set_seed(42)
    for i in range(1, 6):
        sent = generate_sentence(uni_counts, n=1, max_len=20, seed=200 + i * 23)
        log(f"  [{i}] {sent}")
    log()

    # 4. Laplace-Smoothed Bigram Model
    log("---------------------------------------------------------------------------")
    log("4. LAPLACE-SMOOTHED BIGRAM MODEL (Add-One Distribution)")
    log("---------------------------------------------------------------------------")
    # Generate under Laplace smoothed transition
    bi_followers = {}
    for (w1, w2), c in bi_counts.items():
        if w2 != BOS:
            bi_followers.setdefault(w1, []).append((w2, c))

    set_seed(42)
    for i in range(1, 6):
        curr = BOS
        tokens = []
        for _ in range(25):
            followers = bi_followers.get(curr, [])
            if not followers:
                break
            words, counts = zip(*followers)
            # Add-1 Laplace weights
            smoothed_weights = [c + 1 for c in counts]
            next_word = sample_with_temperature(words, smoothed_weights, temperature=0.8)
            if next_word == EOS:
                break
            tokens.append(next_word)
            curr = next_word
        log(f"  [{i}] {' '.join(tokens)}")
    log()

    # Save to file
    out_file = Path(__file__).resolve().parent / "generated_sentences.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"[OK] Output saved to: {out_file}")


if __name__ == "__main__":
    main()
