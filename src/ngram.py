import math
import random
from collections import Counter, defaultdict

from .preprocess import BOS, EOS, preprocess_text


def ngrams(tokens, n):
    """Generate n-tuples of adjacent tokens."""
    if n < 1:
        raise ValueError("n harus >=1")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


# Task 5 required function
def generate_ngrams(tokens, n):
    """Generate n-grams (unigram, bigram, trigram) as a list of tuples."""
    return list(ngrams(tokens, n))


# Task 3 required function
def add_sentence_boundaries(tokens):
    """Task 3: Add <s> at the beginning and </s> at the end of every sentence."""
    return [BOS] + list(tokens) + [EOS]


def add_boundaries(tokens, n=2):
    """Add (n-1) BOS tokens at the start and 1 EOS token at the end."""
    if n < 1:
        raise ValueError("n harus >=1")
    return [BOS] * (n - 1) + list(tokens) + [EOS]


def events(tokens, n):
    """Yield all n-gram events for a sentence after padding with sentence boundaries."""
    yield from ngrams(add_boundaries(tokens, n), n)


def count_ngrams(data, n=None):
    """Task 6: Count n-grams, context (n-1 grams), and follower token distributions.
    
    If n is None, directly counts the provided iterable of n-gram tuples.
    If n is specified, extracts events from sentences and returns (counts, context_counts, followers).
    """
    if n is None:
        return Counter(data)
    counts = Counter()
    context_counts = Counter()
    followers = defaultdict(Counter)
    for sentence in data:
        for gram in events(sentence, n):
            counts[gram] += 1
            ctx = gram[:-1]
            target = gram[-1]
            context_counts[ctx] += 1
            followers[ctx][target] += 1
    return counts, context_counts, followers


class NGramCounter:
    """Convenience class to store and query n-gram counts and follower distributions

    from a corpus.
    """

    def __init__(self, n):
        if n < 1:
            raise ValueError("n harus >=1")
        self.n = n
        self.counts = Counter()
        self.context_counts = Counter()
        self.followers = defaultdict(Counter)

    def fit(self, sentences):
        counts, context_counts, followers = count_ngrams(sentences, self.n)
        self.counts = counts
        self.context_counts = context_counts
        self.followers = followers
        return self

    def get_count(self, gram):
        return self.counts[tuple(gram)]

    def get_context_count(self, context):
        return self.context_counts[tuple(context)]

    def get_followers(self, context):
        return self.followers[tuple(context)]


# Task 7 required function
def calculate_mle_probability(ngram, ngram_counts, context_counts):
    """Task 7 / Section 22: Calculate MLE probability for an n-gram.

    Formula:
        P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})
        For general n-gram:
        P(w_i | w_{i-n+1}^{i-1}) = Count(ngram) / Count(context)

    Args:
        ngram (tuple): The n-gram tuple, e.g. ('of', 'the').
        ngram_counts (Mapping): Frequency counts of n-grams.
        context_counts (Mapping): Frequency counts of contexts ((n-1)-grams).

    Returns:
        float: Maximum Likelihood Estimation probability (0.0 if context count is 0).
    """
    ngram = tuple(ngram)
    context = ngram[:-1]
    c_ngram = ngram_counts.get(ngram, 0)

    # Support context stored as single token or as tuple
    c_ctx = context_counts.get(context, 0)
    if c_ctx == 0 and len(context) == 1:
        c_ctx = context_counts.get(context[0], 0)

    return c_ngram / c_ctx if c_ctx > 0 else 0.0


# Task 9 required function
def predict_next_word(context, ngram_counts, context_counts=None, top_k=5):
    """Task 9 / Section 22: Predict the most probable next word(s) given a context.

    Args:
        context (str or tuple or list): Preceding word(s), e.g. "mr" or ("it", "is").
        ngram_counts (Mapping): Frequency counts of n-grams (e.g. bigram_counts or trigram_counts).
        context_counts (Mapping, optional): Frequency counts of contexts ((n-1)-grams).
            If omitted or not found, computed automatically as sum of follower counts.
        top_k (int, optional): Number of top predictions to return (default: 5). If None, returns all.

    Returns:
        list of tuple: List of (word, probability) sorted descending by probability.
                       Returns empty list [] if context has no observations.
    """
    if isinstance(context, str):
        context = tuple(context.strip().split())
    else:
        context = tuple(context)

    # Filter all n-grams starting with context
    target_len = len(context) + 1
    candidates = {}
    for gram, count in ngram_counts.items():
        if len(gram) == target_len and gram[:-1] == context:
            candidates[gram[-1]] = count

    if not candidates:
        return []

    # Determine total context count
    c_ctx = 0
    if context_counts is not None:
        c_ctx = context_counts.get(context, 0)
        if c_ctx == 0 and len(context) == 1:
            c_ctx = context_counts.get(context[0], 0)

    if c_ctx == 0:
        c_ctx = sum(candidates.values())

    if c_ctx == 0:
        return []

    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]

    return [(word, count / c_ctx) for word, count in sorted_candidates]


# Task 10 required function
def sentence_probability(
    sentence,
    ngram_counts,
    context_counts=None,
    n=2,
    smoothing_fn=None,
    return_breakdown=False,
):
    """Task 10 / Section 22: Calculate sentence probability under n-gram model.

    Formula:
        P(sentence) = Product of P(w_i | w_{i-n+1}^{i-1})
    """
    if isinstance(sentence, str):
        tokens = preprocess_text(sentence)
    else:
        tokens = list(sentence)

    if not tokens:
        return (0.0, []) if return_breakdown else 0.0

    if tokens[0] != BOS:
        tokens = [BOS] * (n - 1) + tokens
    if tokens[-1] != EOS:
        tokens = tokens + [EOS]

    grams = list(ngrams(tokens, n))
    total_prob = 1.0
    breakdown = []

    for gram in grams:
        if smoothing_fn is not None:
            prob = smoothing_fn(gram, ngram_counts, context_counts)
        else:
            prob = calculate_mle_probability(gram, ngram_counts, context_counts)

        total_prob *= prob
        breakdown.append((gram, prob))

    if return_breakdown:
        return total_prob, breakdown
    return total_prob


# Task 11 required function
def sentence_log_probability(
    sentence,
    ngram_counts,
    context_counts=None,
    n=2,
    smoothing_fn=None,
    base=math.e,
    return_breakdown=False,
):
    """Task 11 / Section 22: Calculate log probability to prevent numerical underflow.

    Formula:
        log P(sentence) = Sum of log P(w_i | w_{i-n+1}^{i-1})
    """
    if isinstance(sentence, str):
        tokens = preprocess_text(sentence)
    else:
        tokens = list(sentence)

    if not tokens:
        return (-float("inf"), []) if return_breakdown else -float("inf")

    if tokens[0] != BOS:
        tokens = [BOS] * (n - 1) + tokens
    if tokens[-1] != EOS:
        tokens = tokens + [EOS]

    grams = list(ngrams(tokens, n))
    log_sum = 0.0
    is_zero = False
    breakdown = []

    for gram in grams:
        if smoothing_fn is not None:
            prob = smoothing_fn(gram, ngram_counts, context_counts)
        else:
            prob = calculate_mle_probability(gram, ngram_counts, context_counts)

        if prob <= 0.0:
            is_zero = True
            log_p = -float("inf")
        else:
            log_p = math.log2(prob) if base == 2 else math.log(prob)

        if not is_zero:
            log_sum += log_p

        breakdown.append((gram, prob, log_p))

    final_log = -float("inf") if is_zero else log_sum
    if return_breakdown:
        return final_log, breakdown
    return final_log


# Task 13 required function
def laplace_probability(ngram, ngram_counts, context_counts=None, vocab_size=None):
    """Task 13 / Section 22: Calculate Add-1 / Laplace smoothed probability.

    Formula:
        P_Laplace(w_i | context) = (Count(context, w_i) + 1) / (Count(context) + V)
    """
    ngram = tuple(ngram)
    context = ngram[:-1]
    c_ngram = ngram_counts.get(ngram, 0)

    c_ctx = 0
    if context_counts is not None:
        c_ctx = context_counts.get(context, 0)
        if c_ctx == 0 and len(context) == 1:
            c_ctx = context_counts.get(context[0], 0)

    if c_ctx == 0:
        target_len = len(ngram)
        c_ctx = sum(
            count
            for g, count in ngram_counts.items()
            if len(g) == target_len and g[:-1] == context
        )

    v = float(vocab_size) if vocab_size is not None else 1.0
    return (c_ngram + 1.0) / (c_ctx + v)


# Task 15 required function
def calculate_perplexity(
    test_sentences,
    ngram_counts,
    context_counts=None,
    vocab_size=None,
    n=2,
    smoothing=True,
):
    """Task 15 / Section 22: Calculate perplexity on test corpus.

    Formula:
        Perplexity = exp(- (1/N) * sum(ln P(w_i | context)))
    """
    total_log_prob = 0.0
    total_words = 0

    for sent in test_sentences:
        if isinstance(sent, str):
            tokens = preprocess_text(sent)
        else:
            tokens = list(sent)

        if not tokens:
            continue

        if tokens[0] != BOS:
            tokens = [BOS] * (n - 1) + tokens
        if tokens[-1] != EOS:
            tokens = tokens + [EOS]

        grams = list(ngrams(tokens, n))
        for gram in grams:
            if smoothing:
                p = laplace_probability(gram, ngram_counts, context_counts, vocab_size)
            else:
                p = calculate_mle_probability(gram, ngram_counts, context_counts)

            if p <= 0.0:
                return float("inf")

            total_log_prob += math.log(p)
            total_words += 1

    if total_words == 0:
        return float("inf")

    return math.exp(-total_log_prob / total_words)


def sample_with_temperature(words, weights, temperature=1.0, rng=None):
    """Sample a token from words with weights scaled by temperature.

    Args:
        words (Sequence): Candidate tokens.
        weights (Sequence[float]): Raw counts or probabilities for candidates.
        temperature (float): Temperature scaling parameter (> 0).
            T <= 0.05: Greedy selection (argmax / mode).
            T = 1.0: Standard probability distribution.
            T > 1.0: Flatter / higher entropy distribution (more creative).
        rng (Random, optional): Random instance or module.

    Returns:
        Token selected according to temperature-scaled probabilities.
    """
    if rng is None:
        rng = random

    if len(words) == 1:
        return words[0]

    if temperature <= 0.05:
        best_idx = max(range(len(weights)), key=lambda i: weights[i])
        return words[best_idx]

    inv_t = 1.0 / max(temperature, 0.01)
    scaled_weights = [math.pow(w, inv_t) for w in weights]
    return rng.choices(words, weights=scaled_weights, k=1)[0]


# Task 16 required function
def generate_sentence(
    ngram_counts,
    context_counts=None,
    n=2,
    max_len=30,
    seed=None,
    temperature=1.0,
):
    """Task 16 / Section 22: Generate a sentence by sampling from n-gram distribution.

    Args:
        ngram_counts (Mapping): Frequency counts of n-grams.
        context_counts (Mapping, optional): Context counts.
        n (int, optional): N-gram order (1: unigram, 2: bigram, 3: trigram). Default: 2.
        max_len (int, optional): Maximum length of sentence. Default: 30.
        seed (int, optional): Random seed. Default: None.
        temperature (float, optional): Sampling temperature. Default: 1.0.

    Returns:
        str: Generated sentence text.
    """
    rng = random.Random(seed) if seed is not None else random

    if n == 1:
        candidates = [
            (g[0], c)
            for g, c in ngram_counts.items()
            if len(g) == 1 and g[0] not in (BOS, EOS)
        ]
        if not candidates:
            return ""
        words, weights = zip(*candidates)
        generated = [
            sample_with_temperature(words, weights, temperature, rng)
            for _ in range(min(max_len, 15))
        ]
        return " ".join(generated)

    elif n == 2:
        current_word = BOS
        generated = []
        for _ in range(max_len):
            followers = [
                (g[1], c)
                for g, c in ngram_counts.items()
                if len(g) == 2 and g[0] == current_word and g[1] != BOS
            ]
            if not followers:
                break
            words, weights = zip(*followers)
            next_word = sample_with_temperature(words, weights, temperature, rng)
            if next_word == EOS:
                break
            generated.append(next_word)
            current_word = next_word
        return " ".join(generated)

    elif n == 3:
        generated = []
        first_followers = [
            (g[1], c)
            for g, c in ngram_counts.items()
            if len(g) >= 2 and g[0] == BOS and g[1] not in (BOS, EOS)
        ]
        if not first_followers:
            return ""
        words, weights = zip(*first_followers)
        w1 = sample_with_temperature(words, weights, temperature, rng)
        generated.append(w1)
        prev_context = (BOS, w1)

        for _ in range(max_len - 1):
            followers = [
                (g[2], c)
                for g, c in ngram_counts.items()
                if len(g) == 3 and g[:2] == prev_context and g[2] != BOS
            ]
            if not followers:
                followers = [
                    (g[1], c)
                    for g, c in ngram_counts.items()
                    if len(g) == 2 and g[0] == prev_context[1] and g[1] != BOS
                ]
            if not followers:
                break
            words, weights = zip(*followers)
            next_word = sample_with_temperature(words, weights, temperature, rng)
            if next_word == EOS:
                break
            generated.append(next_word)
            prev_context = (prev_context[1], next_word)
        return " ".join(generated)

    else:
        raise ValueError("n harus 1, 2, atau 3")


