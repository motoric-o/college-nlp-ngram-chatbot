"""N-grams generation, sentence boundaries, and n-gram counting."""

from collections import Counter, defaultdict

from .preprocess import BOS, EOS


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
