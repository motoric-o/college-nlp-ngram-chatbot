"""Implementation and verification script for Task 7 and beyond of the assignment PDF:
'Building a Simple N-gram Language Model from Scratch in Python'.

Covers:
  Task 7: Calculate MLE Probability
  Task 8: Compare Unigram, Bigram, and Trigram
  (Future Tasks 9 to 17 will be appended here as we advance)

Usage:
    python tasks_7_plus.py
    python tasks_7_plus.py --quick
"""

import sys
import argparse
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
    preprocess_text,
    split_corpus,
)
from src.ngram import (
    add_sentence_boundaries,
    calculate_mle_probability,
    count_ngrams,
    generate_ngrams,
)


def print_task_header(task_num, title):
    print("\n" + "=" * 70)
    print(f"Task {task_num}: {title}")
    print("=" * 70)


def prepare_ngram_dataset(corpus_name="pride", quick=False, seed=42):
    """Load, preprocess, boundary-pad, split, and generate n-grams for tasks 7+."""
    set_seed(seed)

    if quick:
        print("    [--quick mode: using sample corpus sentences]")
        raw_sents = [
            ["It", "is", "a", "truth", "universally", "acknowledged", ",", "that", "a", "single", "man", "."],
            ["The", "cat", "is", "sitting", "on", "the", "mat", "."],
            ["The", "cat", "sleeps", "."],
        ] * 1000
    else:
        raw_sents = load_corpus(corpus_name)

    clean_sents = [preprocess_text(s) for s in raw_sents]
    clean_sents = [s for s in clean_sents if s]

    bounded_sents = [add_sentence_boundaries(s) for s in clean_sents]
    train_sents, test_sents, _, _ = split_corpus(bounded_sents, train_ratio=0.8, seed=seed)

    all_unigrams = [u for s in train_sents for u in generate_ngrams(s, 1)]
    all_bigrams = [b for s in train_sents for b in generate_ngrams(s, 2)]
    all_trigrams = [t for s in train_sents for t in generate_ngrams(s, 3)]

    unigram_counts = count_ngrams(all_unigrams)
    bigram_counts = count_ngrams(all_bigrams)
    trigram_counts = count_ngrams(all_trigrams)
    context_counts = Counter(b[0] for b in all_bigrams)

    print(f"    - Training sentences: {len(train_sents):,} | Test sentences: {len(test_sents):,}")
    print(
        f"    - Generated tokens: {len(all_unigrams):,} unigrams, "
        f"{len(all_bigrams):,} bigrams, {len(all_trigrams):,} trigrams"
    )

    return {
        "raw_sents": raw_sents,
        "clean_sents": clean_sents,
        "train_sents": train_sents,
        "test_sents": test_sents,
        "all_unigrams": all_unigrams,
        "all_bigrams": all_bigrams,
        "all_trigrams": all_trigrams,
        "unigram_counts": unigram_counts,
        "bigram_counts": bigram_counts,
        "trigram_counts": trigram_counts,
        "context_counts": context_counts,
    }


def task_7_calculate_mle_probability(bigram_counts, context_counts):
    """Task 7: Calculate MLE probability for bigrams with formula and examples."""
    print_task_header(7, "Calculate MLE Probability")
    print("Implement the Maximum Likelihood Estimation formula.")
    print("For a bigram:")
    print("  P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})\n")
    print("This is the fundamental probability estimation method presented in the chapter.\n")
    print("For example:")
    print("  Count(the, cat) = 25")
    print("  Count(the)      = 500")
    print("  Therefore:")
    print("  P(cat | the) = 25/500 = 0.05\n")

    # Example bigrams as requested: "Students should calculate probabilities for at least five example bigrams."
    examples = [
        ("of", "the"),
        ("to", "be"),
        ("in", "the"),
        ("mr", "darcy"),
        (BOS, "i"),
        ("unusual", "machine"),  # Demonstrating unseen combination (zero probability)
    ]

    print("Expected output\n")
    header = f"{'Bigram':<25} {'Count':<8} {'Context Count':<15} {'Probability':<12}"
    print(header)
    print("-" * len(header))
    for w1, w2 in examples:
        ngram = (w1, w2)
        c = bigram_counts.get(ngram, 0)
        c_ctx = context_counts.get(w1, 0)
        prob = calculate_mle_probability(ngram, bigram_counts, context_counts)
        bigram_str = f"{w1} -> {w2}"
        print(f"{bigram_str:<25} {c:<8,} {c_ctx:<15,} {prob:.4f}")

    return examples


def task_8_compare_models(all_unigrams, all_bigrams, all_trigrams):
    """Task 8: Compare unique unigrams, bigrams, and trigrams and explain sparsity."""
    print_task_header(8, "Compare Unigram, Bigram, and Trigram")
    print("Calculate the number of unique:")
    print("  * unigrams,")
    print("  * bigrams,")
    print("  * trigrams.\n")

    n_unique_unigrams = len(set(all_unigrams))
    n_unique_bigrams = len(set(all_bigrams))
    n_unique_trigrams = len(set(all_trigrams))

    print("Expected output\n")
    table_header = f"{'Model':<25} {'Unique N-grams':>15}"
    print(table_header)
    print("-" * len(table_header))
    print(f"{'Unigram':<25} {n_unique_unigrams:>15,}")
    print(f"{'Bigram':<25} {n_unique_bigrams:>15,}")
    print(f"{'Trigram':<25} {n_unique_trigrams:>15,}")

    v_size = n_unique_unigrams
    possible_unigrams = v_size
    possible_bigrams = v_size**2
    possible_trigrams = v_size**3

    bigram_ratio = (n_unique_bigrams / possible_bigrams) * 100 if possible_bigrams else 0
    trigram_ratio = (n_unique_trigrams / possible_trigrams) * 100 if possible_trigrams else 0

    print("\n" + "-" * 70)
    print("Analysis and Observations")
    print("-" * 70)
    print(
        "\n1. Why does the number of possible N-grams increase rapidly when N increases?\n"
        f"   - The theoretical search space scales exponentially as |V|^N, where |V| is vocabulary size.\n"
        f"   - For our training vocabulary (|V| = {v_size:,}):\n"
        f"     * Possible Unigrams (|V|^1) : {possible_unigrams:,}\n"
        f"     * Possible Bigrams  (|V|^2) : {possible_bigrams:,} (~{possible_bigrams / 1e6:.1f} million)\n"
        f"     * Possible Trigrams (|V|^3) : {possible_trigrams:,} (~{possible_trigrams / 1e9:.1f} billion)\n"
    )
    print(
        "2. The Sparsity Issue (Zero-Frequency Problem):\n"
        f"   - Although theoretical combinations explode exponentially, the training text is finite\n"
        f"     (~{len(all_bigrams):,} bigram tokens in our training set).\n"
        f"   - Consequently, only a tiny fraction of possible N-grams is ever observed:\n"
        f"     * Observed Bigrams  : {n_unique_bigrams:,} / {possible_bigrams:,} ({bigram_ratio:.4f}%)\n"
        f"     * Observed Trigrams : {n_unique_trigrams:,} / {possible_trigrams:,} ({trigram_ratio:.7f}%)\n"
        "   - Higher-order models capture richer local context, but suffer from extreme sparsity: most\n"
        "     grammatically valid word combinations never appear in any finite training corpus.\n"
        "   - Under pure Maximum Likelihood Estimation (MLE), any unseen N-gram has Count = 0, giving\n"
        "     P(w_i | context) = 0. When computing sentence probability by multiplying conditional\n"
        "     probabilities, a single unseen N-gram causes the entire sentence probability to become 0.\n"
        "   - This directly motivates smoothing techniques (such as Laplace/Add-1 smoothing in Task 13)\n"
        "     to assign non-zero probability mass to unseen events."
    )

    return {
        "unigrams": n_unique_unigrams,
        "bigrams": n_unique_bigrams,
        "trigrams": n_unique_trigrams,
    }


def main():
    parser = argparse.ArgumentParser(description="Tasks 7 and Beyond of N-gram Language Model Assignment")
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
    print("  Pipeline Implementation: Tasks 7 and Beyond")
    print("#" * 70 + "\n")

    # Step A: Prepare dataset from earlier tasks (Tasks 1 to 5)
    data = prepare_ngram_dataset(corpus_name=args.corpus, quick=args.quick)

    # Task 7: Calculate MLE Probability
    task_7_calculate_mle_probability(data["bigram_counts"], data["context_counts"])

    # Task 8: Compare Unigram, Bigram, and Trigram
    task_8_compare_models(data["all_unigrams"], data["all_bigrams"], data["all_trigrams"])

    print("\n" + "=" * 70)
    print(">> TASKS 7 TO 8 SUCCESSFULLY COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
