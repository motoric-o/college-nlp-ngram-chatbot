"""Implementation and verification script for Tasks 1 to 6 of the assignment PDF:
'Building a Simple N-gram Language Model from Scratch in Python'.

Covers:
  Task 1: Load and Inspect the Corpus
  Task 2: Text Preprocessing
  Task 3: Add Sentence Boundary Symbols
  Task 4: Split the Corpus
  Task 5: Generate N-grams
  Task 6: Count N-grams

Usage:
    python test_all.py
    python test_all.py --quick
"""

import sys
import argparse
import random
from collections import Counter
from pathlib import Path

# Add project root to sys.path so 'src' is always importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import load_corpus, set_seed
from src.preprocess import (
    BOS,
    EOS,
    preprocess,
    preprocess_text,
    split_corpus,
    tokenize,
)
from src.ngram import (
    add_sentence_boundaries,
    count_ngrams,
    generate_ngrams,
)


def print_task_header(task_num: int, task_name: str):
    banner = "=" * 70
    print(f"\n{banner}")
    print(f"Task {task_num}: {task_name}")
    print(banner)


def task_1_load_and_inspect_corpus(corpus_name="pride", quick=False):
    print_task_header(1, "Load and Inspect the Corpus")
    set_seed(42)

    if quick:
        print("[--quick mode: using sample corpus sentences]")
        raw_sentences = [
            ["It", "is", "a", "truth", "universally", "acknowledged", ",", "that", "a", "single", "man", "."],
            ["The", "cat", "is", "sitting", "on", "the", "mat", "."],
            ["The", "cat", "sleeps", "."],
        ] * 1000
    else:
        raw_sentences = load_corpus(corpus_name)

    # Basic corpus metrics
    num_sentences = len(raw_sentences)
    all_tokens = [token for s in raw_sentences for token in s]
    num_tokens = len(all_tokens)
    vocabulary = set(all_tokens)
    vocab_size = len(vocabulary)
    avg_sentence_len = num_tokens / num_sentences if num_sentences else 0.0

    # Five most frequent words (filtering punctuation for word frequencies)
    word_tokens = [w.lower() for w in all_tokens if w.isalpha()]
    word_counts = Counter(word_tokens)
    top_5_words = word_counts.most_common(5)

    print("Expected output\n")
    print(f"Number of sentences : {num_sentences:,}")
    print(f"Number of tokens    : {num_tokens:,}")
    print(f"Vocabulary size     : {vocab_size:,}")
    print(f"Average sentence    : {avg_sentence_len:.1f} words")
    print("\nTop 5 words:")
    for word, count in top_5_words:
        print(f"{word:<10} {count:,}")

    return raw_sentences


def task_2_text_preprocessing(raw_sentences):
    print_task_header(2, "Text Preprocessing")
    print("Preprocess requirements:")
    print("  1. Convert text to lowercase")
    print("  2. Remove unnecessary characters")
    print("  3. Tokenize the text")
    print("  4. Remove empty tokens\n")

    # Expected output: show at least three examples of Original sentence and Processed tokens
    examples = [
        "The cat is sitting on the MAT.",  # Exact example from PDF
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "\"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\"",
        "Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley!",
    ]

    print("Expected output (Examples of Original sentence and Processed tokens):")
    for i, ex in enumerate(examples, start=1):
        processed = preprocess_text(ex)
        print(f"\nExample {i}:")
        print(f"Original:\n\"{ex}\"")
        print(f"After preprocessing:\n{processed}")

    # Process the loaded corpus
    clean_sentences, dropped = preprocess(raw_sentences)
    print(f"\n[Corpus summary] Total cleaned sentences: {len(clean_sentences):,} (dropped empty: {dropped})")
    return clean_sentences


def task_3_add_sentence_boundary_symbols(clean_sentences):
    print_task_header(3, "Add Sentence Boundary Symbols")
    print(f"Add {BOS} at the beginning and {EOS} at the end of every sentence.\n")

    # Example from PDF
    pdf_example_tokens = ["the", "cat", "sleeps"]
    pdf_bounded = add_sentence_boundaries(pdf_example_tokens)

    print("For example:")
    print("Original:")
    print("The cat sleeps.")
    print("Processed:")
    print(" ".join(pdf_bounded))

    print("\nCorpus examples with boundaries:")
    sample_corpus_examples = [
        ["it", "is", "a", "truth", "universally", "acknowledged"],
        ["my", "dear", "mr", "bennet", "said", "his", "lady", "to", "him", "one", "day"],
    ]
    for i, ex in enumerate(sample_corpus_examples, start=1):
        bounded = add_sentence_boundaries(ex)
        print(f"\nExample {i}:")
        print(f"Original:  {' '.join(ex)}.")
        print(f"Processed: {' '.join(bounded)}")

    # Add boundaries to all sentences in corpus
    bounded_corpus = [add_sentence_boundaries(s) for s in clean_sentences]
    return bounded_corpus


def task_4_split_the_corpus(bounded_sentences, train_ratio=0.8, seed=42):
    print_task_header(4, "Split the Corpus")
    print("Split the corpus into:")
    print(f"  * {round(train_ratio * 100)}% training data")
    print(f"  * {round((1 - train_ratio) * 100)}% test data")
    print(f"Fixed random seed: random.seed({seed})\n")

    train_sents, test_sents, train_ids, test_ids = split_corpus(
        bounded_sentences, train_ratio=train_ratio, seed=seed
    )

    print("Expected output\n")
    print(f"Total sentences : {len(bounded_sentences):,}")
    print(f"Training        : {len(train_sents):,}")
    print(f"Testing         : {len(test_sents):,}")

    return train_sents, test_sents


def task_5_generate_ngrams(train_sentences):
    print_task_header(5, "Generate N-grams")
    print("Implement a function that generates: unigram, bigram, trigram.\n")

    # PDF Demonstration example
    cat_sample = [BOS, "the", "cat", "sleeps", EOS]
    unigrams_ex = generate_ngrams(cat_sample, 1)
    bigrams_ex = generate_ngrams(cat_sample, 2)
    trigrams_ex = generate_ngrams(cat_sample, 3)

    print(f"For example:")
    print(" ".join(cat_sample))
    print("\nUnigrams (shown vertically as in PDF):")
    for u in unigrams_ex:
        print(u[0])
    print("\nBigrams:")
    for b in bigrams_ex:
        print(b)
    print("\nTrigrams:")
    for t in trigrams_ex:
        print(t)

    # Generate from training corpus
    all_unigrams = [u for s in train_sentences for u in generate_ngrams(s, 1)]
    all_bigrams = [b for s in train_sentences for b in generate_ngrams(s, 2)]
    all_trigrams = [t for s in train_sentences for t in generate_ngrams(s, 3)]

    print("\nExpected output")
    print("Display:")
    print("First 10 unigrams:")
    for u in all_unigrams[:10]:
        print(f"  {u}")

    print("\nFirst 10 bigrams:")
    for b in all_bigrams[:10]:
        print(f"  {b}")

    print("\nFirst 10 trigrams:")
    for t in all_trigrams[:10]:
        print(f"  {t}")

    return all_unigrams, all_bigrams, all_trigrams


def task_6_count_ngrams(all_bigrams):
    print_task_header(6, "Count N-grams")
    print("Use Python's Counter or a dictionary.")
    print("Example: bigram_counts = Counter(bigrams)\n")

    bigram_counts = count_ngrams(all_bigrams)

    print("Expected output")
    print("Top 10 bigrams:")
    top_10 = bigram_counts.most_common(10)
    for bigram, count in top_10:
        print(f"{str(bigram):<25} {count:,}")

    return bigram_counts


def main():
    parser = argparse.ArgumentParser(description="Tasks 1 to 6 of N-gram Language Model Assignment")
    parser.add_argument(
        "--corpus",
        default="pride",
        choices=["pride", "pride_and_prejudice", "brown", "reuters"],
        help="Corpus to use (default: 'pride')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quickly with sample sentences",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  BUILDING A SIMPLE N-GRAM LANGUAGE MODEL FROM SCRATCH IN PYTHON")
    print("  Pipeline Implementation: Tasks 1 to 6")
    print("#" * 70)

    # Task 1: Load and Inspect the Corpus
    raw_sents = task_1_load_and_inspect_corpus(corpus_name=args.corpus, quick=args.quick)

    # Task 2: Text Preprocessing
    clean_sents = task_2_text_preprocessing(raw_sents)

    # Task 3: Add Sentence Boundary Symbols
    bounded_sents = task_3_add_sentence_boundary_symbols(clean_sents)

    # Task 4: Split the Corpus
    train_sents, test_sents = task_4_split_the_corpus(bounded_sents, train_ratio=0.8, seed=42)

    # Task 5: Generate N-grams
    all_unigrams, all_bigrams, all_trigrams = task_5_generate_ngrams(train_sents)

    # Task 6: Count N-grams
    bigram_counts = task_6_count_ngrams(all_bigrams)

    print("\n" + "=" * 70)
    print(">> TASKS 1 TO 6 SUCCESSFULLY COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
