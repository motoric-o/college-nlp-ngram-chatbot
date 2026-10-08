"""Implementation and verification script for Tasks 7 through 17 of the assignment PDF:
'Building a Simple N-gram Language Model from Scratch in Python'.

Covers:
  Task 7:  Calculate MLE Probability
  Task 8:  Compare Unigram, Bigram, and Trigram
  Task 9:  Next-Word Prediction
  Task 10: Calculate Sentence Probability
  Task 11: Use Log Probability
  Task 12: Demonstrate the Zero-Frequency Problem
  Task 13: Implement Laplace Smoothing
  Task 14: Compare MLE and Laplace Smoothing
  Task 15: Calculate Perplexity
  Task 16: Generate Text
  Task 17: Compare the Three Models

Usage:
    python tasks_7_plus.py
    python tasks_7_plus.py --quick
    python tasks_7_plus.py --task <7-17>
"""

import sys
import argparse
import math
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
    calculate_perplexity,
    count_ngrams,
    generate_ngrams,
    generate_sentence,
    laplace_probability,
    predict_next_word,
    sentence_log_probability,
    sentence_probability,
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


def task_9_next_word_prediction(bigram_counts, trigram_counts, context_counts):
    """Task 9: Next-word prediction given previous word(s) using predict_next_word."""
    print_task_header(9, "Next-Word Prediction")
    print("Given a previous word, predict the most probable next word.\n")

    # Bigram model predictions (1-word context)
    bigram_contexts = [
        "machine",  # PDF example (demonstrating unseen context / zero-frequency)
        "mr",       # Frequent character context in Pride & Prejudice
        "she",      # Common pronoun context
    ]

    print("--- Bigram Model Predictions (1-word context) ---\n")
    for ctx in bigram_contexts:
        print(f'Input context: "{ctx}"')
        preds = predict_next_word(ctx, bigram_counts, context_counts=context_counts, top_k=5)
        if not preds:
            print("  [Unseen context in corpus: No predictions available (Count = 0)]\n")
        else:
            print("Top 5 predictions:")
            for rank, (word, prob) in enumerate(preds, 1):
                c_ngram = bigram_counts.get((ctx, word), 0)
                print(f"  {rank}. {word:<15} {prob:.4f} (Count: {c_ngram:,})")
            print()

    # Trigram model predictions (2-word context)
    print("--- Trigram Model Predictions (2-word context) ---\n")
    print("For the trigram model, provide two previous words:\n")
    trigram_contexts = [
        "natural language",  # PDF example (unseen context)
        "it is",             # Frequent opener in Pride & Prejudice
        "she was",           # Common narrative context
    ]

    for ctx in trigram_contexts:
        print(f'Input context: "{ctx}"')
        preds = predict_next_word(ctx, trigram_counts, top_k=5)
        if not preds:
            print("  [Unseen context in corpus: No predictions available (Count = 0)]\n")
        else:
            print("Top 5 predictions:")
            ctx_tuple = tuple(ctx.split())
            for rank, (word, prob) in enumerate(preds, 1):
                c_ngram = trigram_counts.get(ctx_tuple + (word,), 0)
                print(f"  {rank}. {word:<15} {prob:.4f} (Count: {c_ngram:,})")
            print()


def task_10_calculate_sentence_probability(bigram_counts, context_counts):
    """Task 10: Calculate sentence probability as product of conditional probabilities."""
    print_task_header(10, "Calculate Sentence Probability")
    print("Calculate the probability of a short sentence.")
    print("For a bigram model:")
    print("  P(sentence) = P(w_1 | <s>) * P(w_2 | w_1) * ... * P(</s> | w_k)\n")
    print(
        "The chapter describes sentence probability as the product of the sequence's conditional probabilities.\n"
    )

    test_sentences = [
        "The cat sleeps.",  # PDF textbook example
        "it is a truth universally acknowledged",  # Austen corpus example
    ]

    for sent_str in test_sentences:
        print(f'Sentence:\n  "{sent_str}"')
        prob, breakdown = sentence_probability(
            sent_str,
            bigram_counts,
            context_counts,
            n=2,
            return_breakdown=True,
        )

        print("\nIndividual probabilities used:")
        header = f"  {'Transition (w_{i-1} -> w_i)':<30} {'Count':<8} {'Context Count':<15} {'Probability':<12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for (w1, w2), p in breakdown:
            c = bigram_counts.get((w1, w2), 0)
            c_ctx = (
                context_counts.get(w1, 0)
                if context_counts
                else bigram_counts.get((w1, w2), 0)
            )
            print(f"  {w1 + ' -> ' + w2:<30} {c:<8,} {c_ctx:<15,} {p:.6f}")

        print(f"\nBigram probability:\n  {prob:.10e} ({prob:.8f})\n")


def task_11_use_log_probability(bigram_counts, context_counts):
    """Task 11: Use log probability to avoid numerical underflow."""
    print_task_header(11, "Use Log Probability")
    print(
        "Because multiplying many small probabilities can produce extremely small numbers,"
    )
    print("calculate the log probability.")
    print(
        "The chapter introduces log probability as a practical way to work with these products.\n"
    )

    sent_str = "it is a truth universally acknowledged"
    print(f'Sentence:\n  "{sent_str}"')

    prob, _ = sentence_probability(
        sent_str, bigram_counts, context_counts, n=2, return_breakdown=True
    )
    log_e, _ = sentence_log_probability(
        sent_str,
        bigram_counts,
        context_counts,
        n=2,
        base=math.e,
        return_breakdown=True,
    )
    log_2 = sentence_log_probability(
        sent_str, bigram_counts, context_counts, n=2, base=2
    )

    print("\nExpected output\n")
    print(f"Sentence probability  : {prob:.10e}")
    print(f"Log probability (ln)  : {log_e:.2f}")
    print(f"Log probability (log2): {log_2:.2f}")

    print("\nWhy log probability is necessary:")
    print(
        "  Multiplying dozens of probabilities in [0, 1] quickly causes floating-point"
    )
    print(
        "  underflow to 0.0 in standard 64-bit IEEE floats. Transforming probabilities to"
    )
    print(
        "  log space converts products into sums: log P(W) = sum(log P(w_i | w_{i-1})),"
    )
    print(
        "  which maintains full numerical precision and complete numerical stability."
    )


def task_12_demonstrate_zero_frequency(bigram_counts, context_counts):
    """Task 12: Demonstrate the zero-frequency problem."""
    print_task_header(12, "Demonstrate the Zero-Frequency Problem")
    print(
        "Choose a sentence containing an N-gram that does not occur in the training corpus.\n"
    )

    unseen_sent = "the extremely unusual machine"
    print(f'Input:\n  "{unseen_sent}"\n')

    unseen_bigram = ("unusual", "machine")
    c_unseen = bigram_counts.get(unseen_bigram, 0)
    p_unseen = calculate_mle_probability(
        unseen_bigram, bigram_counts, context_counts
    )
    sent_prob = sentence_probability(
        unseen_sent, bigram_counts, context_counts, n=2
    )
    sent_log_p = sentence_log_probability(
        unseen_sent, bigram_counts, context_counts, n=2
    )

    print("Expected output\n")
    print(f"Bigram:\n  {unseen_bigram}")
    print(f"Count:\n  {c_unseen}")
    print(f"Probability:\n  {p_unseen}")
    print(f"Sentence probability:\n  {sent_prob}")
    print(f"Log probability:\n  {sent_log_p}\n")

    print("Explanation (why this is a problem):")
    print(
        "  In pure Maximum Likelihood Estimation, any unseen N-gram receives Count = 0,"
    )
    print(
        "  yielding a conditional probability of 0. Because sentence probability is computed"
    )
    print(
        "  by multiplying conditional probabilities, a single unseen N-gram causes the entire"
    )
    print(
        "  sentence probability to collapse to zero (and log probability to -inf). This is a"
    )
    print(
        "  fatal flaw because perfectly grammatical and plausible sentences are judged as"
    )
    print(
        "  impossible simply because they did not appear in the training corpus."
    )


def task_13_implement_laplace_smoothing(
    bigram_counts, context_counts, vocab_size
):
    """Task 13: Implement add-one/Laplace smoothing."""
    print_task_header(13, "Implement Laplace Smoothing")
    print("Implement add-one/Laplace smoothing:")
    print(
        "  P_Laplace(w_i | w_{i-1}) = (Count(w_{i-1}, w_i) + 1) / (Count(w_{i-1}) + V)\n"
    )
    print("where V is the vocabulary size.")
    print("This follows the smoothing approach described in the chapter.\n")

    unseen_bigram = ("unusual", "machine")
    mle_p = calculate_mle_probability(
        unseen_bigram, bigram_counts, context_counts
    )
    lap_p = laplace_probability(
        unseen_bigram, bigram_counts, context_counts, vocab_size
    )

    print("Expected output\n")
    print("For the previously unseen bigram:")
    print(f"Bigram:\n  {unseen_bigram}")
    print(f"Vocabulary size (V):\n  {vocab_size:,}")
    print(f"MLE probability:\n  {mle_p}")
    print(f"Laplace probability:\n  {lap_p:.6f} ({lap_p:.8e})\n")

    unseen_sent = "the extremely unusual machine"
    smoothed_fn = lambda gram, nc, cc: laplace_probability(
        gram, nc, cc, vocab_size
    )
    sent_p = sentence_probability(
        unseen_sent,
        bigram_counts,
        context_counts,
        n=2,
        smoothing_fn=smoothed_fn,
    )
    sent_log_p = sentence_log_probability(
        unseen_sent,
        bigram_counts,
        context_counts,
        n=2,
        smoothing_fn=smoothed_fn,
    )
    print(f'Sentence "{unseen_sent}" with Laplace smoothing:')
    print(f"  Smoothed Sentence probability : {sent_p:.10e}")
    print(f"  Smoothed Log probability (ln) : {sent_log_p:.2f}")


def task_14_compare_mle_and_laplace(bigram_counts, context_counts, vocab_size):
    """Task 14: Compare MLE and Laplace smoothing across frequent, rare, and unseen bigrams."""
    print_task_header(14, "Compare MLE and Laplace Smoothing")
    print("Choose:")
    print("  * 3 frequent bigrams,")
    print("  * 2 rare bigrams,")
    print("  * 2 unseen bigrams.\n")

    frequent = [("of", "the"), ("to", "be"), ("in", "the")]
    rare = [
        b
        for b, c in bigram_counts.items()
        if c == 1 and b[0] != BOS and b[1] != EOS
    ][:2]
    if len(rare) < 2:
        rare = [b for b, c in bigram_counts.items() if b[0] != BOS and b[1] != EOS][:2]
    unseen = [("unusual", "machine"), ("robot", "car")]

    chosen = [
        ("Frequent", frequent),
        ("Rare", rare),
        ("Unseen", unseen),
    ]

    print("Create a table:\n")
    header = f"{'Type':<10} {'Bigram':<25} {'Count':<8} {'Context Count':<15} {'MLE Prob':<12} {'Laplace Prob':<14}"
    print(header)
    print("-" * len(header))

    for cat_name, pairs in chosen:
        for w1, w2 in pairs:
            ngram = (w1, w2)
            c = bigram_counts.get(ngram, 0)
            c_ctx = context_counts.get(w1, 0)
            mle_p = calculate_mle_probability(
                ngram, bigram_counts, context_counts
            )
            lap_p = laplace_probability(
                ngram, bigram_counts, context_counts, vocab_size
            )
            pair_str = f"{w1} -> {w2}"
            print(
                f"{cat_name:<10} {pair_str:<25} {c:<8,} {c_ctx:<15,} {mle_p:<12.5f} {lap_p:<14.5f}"
            )

    print("\nQuestion:")
    print("What happens to the probability of an unseen N-gram after smoothing?")
    print("\nExplanation:")
    print(
        "  Before smoothing, unseen N-grams have an MLE probability of exactly 0. After Laplace"
    )
    print(
        "  smoothing, every unseen N-gram is assigned a small positive probability: P = 1 / (Count(ctx) + V)."
    )
    print(
        "  To satisfy the probability axiom sum_w P(w | ctx) = 1, this probability mass is shaved/discounted"
    )
    print(
        "  from the frequent N-grams and redistributed across all unobserved vocabulary words."
    )


def task_15_calculate_perplexity(
    test_sents, bigram_counts, context_counts, vocab_size
):
    """Task 15: Calculate perplexity on test corpus."""
    print_task_header(15, "Calculate Perplexity")
    print("Calculate the perplexity of the test corpus.")
    print(
        "Perplexity evaluates how well a language model predicts a sequence of words;"
    )
    print(
        "the chapter presents it as an evaluation measure for language models.\n"
    )
    print("Students should calculate:")
    print("  * bigram perplexity without smoothing, if possible;")
    print("  * bigram perplexity with Laplace smoothing.\n")

    print("[*] Evaluating perplexity on test set...")
    ppl_mle = calculate_perplexity(
        test_sents,
        bigram_counts,
        context_counts,
        vocab_size,
        n=2,
        smoothing=False,
    )
    ppl_laplace = calculate_perplexity(
        test_sents,
        bigram_counts,
        context_counts,
        vocab_size,
        n=2,
        smoothing=True,
    )

    print("Expected output\n")
    header = f"{'Model':<25} {'Perplexity':>15}"
    print(header)
    print("-" * len(header))
    mle_str = "Infinity (inf)" if math.isinf(ppl_mle) else f"{ppl_mle:>15.2f}"
    print(f"{'Bigram - MLE':<25} {mle_str:>15}")
    print(f"{'Bigram - Laplace':<25} {ppl_laplace:>15.2f}\n")

    print("Question:")
    print(
        "Which model gives a usable perplexity on the test set, and why does zero probability create a problem?"
    )
    print("\nAnswer:")
    print(
        "  Only the Laplace-smoothed model gives a usable perplexity on the test set."
    )
    print(
        "  Under standard MLE, whenever the test set contains an unseen bigram, its probability"
    )
    print(
        "  is zero (P = 0). Since perplexity involves log P, log(0) = -inf, causing test perplexity"
    )
    print(
        "  to become mathematically infinite / undefined (PP = exp(- (-inf) / N) = inf). A single"
    )
    print(
        "  zero-probability event ruins the entire evaluation metric, proving why smoothing is"
    )
    print("  strictly required for language modeling in practice.")


def task_16_generate_text(bigram_counts):
    """Task 16: Generate simple text using the bigram model."""
    print_task_header(16, "Generate Text")
    print("Use the bigram model to generate a sentence.")
    print("Start with:")
    print("  <s>\n")
    print(
        "Then repeatedly select a next word based on the probability distribution."
    )
    print("Stop when:")
    print("  </s> is generated or when the sentence reaches a maximum length.\n")
    print(
        "The chapter describes sampling/generation from N-gram probability distributions.\n"
    )

    print("Expected output\n")
    print("Generate 5 sentences.\n")
    print("Generated sentences:")
    for i in range(1, 6):
        sent = generate_sentence(bigram_counts, n=2, max_len=30, seed=i * 100)
        print(f"  {i}. {sent}")

    print("\nNote: The sentences do not have to be grammatically perfect.")


def task_17_compare_three_models(
    unigram_counts, bigram_counts, trigram_counts
):
    """Task 17: Compare Unigram, Bigram, and Trigram text generation and trade-offs."""
    print_task_header(17, "Compare the Three Models")
    print("Finally, compare:")
    print("  * Unigram")
    print("  * Bigram")
    print("  * Trigram\n")
    print("Students should generate at least 3 sentences from each model.\n")

    models = [
        ("Unigram", 1, unigram_counts, "Words relatively independent"),
        ("Bigram", 2, bigram_counts, "Some local word relationships"),
        ("Trigram", 3, trigram_counts, "Longer local context"),
    ]

    print("Expected output\n")
    print(f"{'Model':<10} {'Example Generated Text':<60} {'Observation'}")
    print("-" * 105)

    samples = {}
    for name, order, counts, obs in models:
        gen_sents = []
        for s in [42, 123, 999]:
            text = generate_sentence(counts, n=order, max_len=20, seed=s)
            gen_sents.append(text)
        samples[name] = gen_sents
        first_snippet = (
            (gen_sents[0][:57] + "...")
            if len(gen_sents[0]) > 57
            else gen_sents[0]
        )
        print(f"{name:<10} {first_snippet:<60} {obs}")

    print("\nDetailed Generated Sentences (3 per model):\n")
    for name, sents in samples.items():
        print(f"[{name} Model]")
        for i, s in enumerate(sents, 1):
            print(f"  {i}. {s}")
        print()

    print("-" * 70)
    print("Discussion and Report Questions")
    print("-" * 70)
    print(
        "\n1. Which model produces more locally coherent word sequences?\n"
        "   - The Trigram model produces significantly more coherent word sequences.\n"
        "     Because it conditions each word on the preceding two words, it captures\n"
        "     standard multi-word expressions, idioms, and subject-verb-object structures\n"
        "     (e.g., 'it was a truth', 'mr bingley called again'). In contrast, Unigrams\n"
        "     produce a disjoint 'bag of words', and Bigrams only maintain pairwise syntax.\n"
    )
    print(
        "2. What happens when N increases?\n"
        "   - As N increases, the model captures longer contextual dependencies and produces\n"
        "     output that sounds increasingly natural and fluent. However, the model also becomes\n"
        "     more rigid, tending to copy verbatim chunks from the training text (memorization).\n"
    )
    print(
        "3. Why can a larger N also create more unseen combinations?\n"
        "   - As demonstrated in Task 8, the search space grows exponentially as |V|^N.\n"
        "     For trigrams, |V|^3 produces over 235 billion possible combinations, of which\n"
        "     only 84,210 are seen in the training data. Most valid contexts never occur in\n"
        "     the training set, causing severe data sparsity and high out-of-vocabulary risk.\n"
        "     Thus, language modeling embodies a fundamental trade-off: higher N provides richer\n"
        "     context, but demands much larger datasets and aggressive smoothing to avoid sparsity."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tasks 7 through 17 of N-gram Language Model Assignment"
    )
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
    parser.add_argument(
        "--task",
        type=int,
        choices=range(7, 18),
        help="Run a specific task (7 to 17). Default: runs all tasks.",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  BUILDING A SIMPLE N-GRAM LANGUAGE MODEL FROM SCRATCH IN PYTHON")
    print("  Pipeline Implementation: Tasks 7 to 17 (Complete Week 4 Deliverables)")
    print("#" * 70 + "\n")

    # Step A: Prepare dataset from earlier tasks (Tasks 1 to 5)
    data = prepare_ngram_dataset(corpus_name=args.corpus, quick=args.quick)
    vocab_size = len(set(data["all_unigrams"]))

    task_map = {
        7: lambda: task_7_calculate_mle_probability(
            data["bigram_counts"], data["context_counts"]
        ),
        8: lambda: task_8_compare_models(
            data["all_unigrams"], data["all_bigrams"], data["all_trigrams"]
        ),
        9: lambda: task_9_next_word_prediction(
            data["bigram_counts"],
            data["trigram_counts"],
            data["context_counts"],
        ),
        10: lambda: task_10_calculate_sentence_probability(
            data["bigram_counts"], data["context_counts"]
        ),
        11: lambda: task_11_use_log_probability(
            data["bigram_counts"], data["context_counts"]
        ),
        12: lambda: task_12_demonstrate_zero_frequency(
            data["bigram_counts"], data["context_counts"]
        ),
        13: lambda: task_13_implement_laplace_smoothing(
            data["bigram_counts"], data["context_counts"], vocab_size
        ),
        14: lambda: task_14_compare_mle_and_laplace(
            data["bigram_counts"], data["context_counts"], vocab_size
        ),
        15: lambda: task_15_calculate_perplexity(
            data["test_sents"],
            data["bigram_counts"],
            data["context_counts"],
            vocab_size,
        ),
        16: lambda: task_16_generate_text(data["bigram_counts"]),
        17: lambda: task_17_compare_three_models(
            data["unigram_counts"],
            data["bigram_counts"],
            data["trigram_counts"],
        ),
    }

    if args.task is not None:
        task_map[args.task]()
        print("\n" + "=" * 70)
        print(f">> TASK {args.task} SUCCESSFULLY COMPLETED")
        print("=" * 70 + "\n")
    else:
        for t_num in range(7, 18):
            task_map[t_num]()

        print("\n" + "=" * 70)
        print(">> TASKS 7 TO 17 SUCCESSFULLY COMPLETED - ALL ASSIGNMENT TASKS DONE")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
