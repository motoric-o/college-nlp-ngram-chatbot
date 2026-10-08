"""N-Gram Chatbot Architecture and Implementations.

Provides:
  - BaseChatbot: Foundation class managing corpus data, n-gram tables, and temperature settings.
  - KeywordSeededChatbot: Conversational chatbot that extracts topic keywords from user messages
    and generates conditioned responses using temperature-scaled N-gram sampling.
  - CHATBOT_REGISTRY: Factory dictionary to easily register and instantiate different chatbot classes.
"""

import math
import random
import re
from typing import Dict, List, Optional, Tuple

from .data import load_corpus, set_seed
from .ngram import (
    BOS,
    EOS,
    add_sentence_boundaries,
    count_ngrams,
    generate_ngrams,
    sample_with_temperature,
)
from .preprocess import preprocess_text, split_corpus

# Common English stopwords to filter when identifying topical keywords
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "what", "which", "who", "whom", "this",
    "that", "these", "those", "how", "why", "where", "when", "there", "here",
    "tell", "say", "think", "please", "know", "hello", "hi", "hey", "so", "as",
    "of", "if", "then", "just", "well", "like", "much", "very", "any", "some",
    "see", "seen", "look", "talk", "give", "good", "morning", "evening", "day",
}

# Proper nouns in Austen's Pride and Prejudice to capitalize correctly in output
AUSTEN_PROPER_NOUNS = {
    "mr": "Mr.",
    "mrs": "Mrs.",
    "miss": "Miss",
    "darcy": "Darcy",
    "bingley": "Bingley",
    "elizabeth": "Elizabeth",
    "jane": "Jane",
    "bennet": "Bennet",
    "wickham": "Wickham",
    "collins": "Collins",
    "lydia": "Lydia",
    "catherine": "Catherine",
    "lady": "Lady",
    "netherfield": "Netherfield",
    "pemberley": "Pemberley",
    "longbourn": "Longbourn",
    "hertfordshire": "Hertfordshire",
    "rosings": "Rosings",
    "charlotte": "Charlotte",
    "lucas": "Lucas",
    "gardiner": "Gardiner",
    "fitzwilliam": "Fitzwilliam",
    "georgiana": "Georgiana",
    "bourgh": "de Bourgh",
}


class BaseChatbot:
    """Abstract base class for all N-Gram chatbot architectures."""

    def __init__(
        self,
        corpus_name: str = "pride",
        n: int = 3,
        temperature: float = 0.8,
        max_len: int = 30,
        quick: bool = False,
        seed: Optional[int] = None,
    ):
        self.corpus_name = corpus_name
        self.n = n
        self.temperature = max(0.01, float(temperature))
        self.max_len = max_len
        self.quick = quick
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

        self._load_and_index_corpus()

    def _load_and_index_corpus(self):
        """Load corpus, preprocess, and compute n-gram count models."""
        if self.quick:
            raw_sents = [
                ["It", "is", "a", "truth", "universally", "acknowledged", ",", "that", "a", "single", "man", "."],
                ["The", "cat", "is", "sitting", "on", "the", "mat", "."],
                ["The", "cat", "sleeps", "."],
            ] * 500
        else:
            raw_sents = load_corpus(self.corpus_name)

        clean_sents = [preprocess_text(s) for s in raw_sents]
        clean_sents = [s for s in clean_sents if s]

        bounded_sents = [add_sentence_boundaries(s) for s in clean_sents]
        train_sents, _, _, _ = split_corpus(bounded_sents, train_ratio=0.8, seed=42)

        self.all_unigrams = [u for s in train_sents for u in generate_ngrams(s, 1)]
        self.all_bigrams = [b for s in train_sents for b in generate_ngrams(s, 2)]
        self.all_trigrams = [t for s in train_sents for t in generate_ngrams(s, 3)]

        self.unigram_counts = count_ngrams(self.all_unigrams)
        self.bigram_counts = count_ngrams(self.all_bigrams)
        self.trigram_counts = count_ngrams(self.all_trigrams)

        self.vocabulary = set(
            w for b in self.bigram_counts for w in b if w not in (BOS, EOS)
        )

    def set_temperature(self, temp: float):
        """Update sampling temperature."""
        self.temperature = max(0.01, float(temp))

    def set_model_order(self, n: int):
        """Set N-gram order (2 for bigram, 3 for trigram)."""
        if n not in (2, 3):
            raise ValueError("n must be 2 (bigram) or 3 (trigram)")
        self.n = n

    def format_response(self, tokens: List[str]) -> str:
        """Post-process tokens into clean, capitalized, punctuated Victorian prose."""
        if not tokens:
            return ""

        formatted = []
        for tok in tokens:
            if tok in (BOS, EOS):
                continue
            # Proper noun capitalization
            if tok in AUSTEN_PROPER_NOUNS:
                formatted.append(AUSTEN_PROPER_NOUNS[tok])
            elif tok == "i":
                formatted.append("I")
            else:
                formatted.append(tok)

        text = " ".join(formatted)

        # Fix contraction and punctuation spacing: " ' s" -> "'s", " , " -> ", "
        text = re.sub(r"\s+([',.!?;:])", r"\1", text)
        text = re.sub(r"([A-Za-z])\s+('\w+)", r"\1\2", text)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
            # Ensure closing punctuation
            if text[-1] not in ".!?":
                text += "."

        return text

    def respond(self, user_input: str) -> str:
        """Abstract respond method to be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement respond(user_input)")


class KeywordSeededChatbot(BaseChatbot):
    """Chatbot implementing Approach 2: Keyword-Seeded Response Generation.

    Extracts topical keywords from user queries, anchors the N-gram context to
    that topic, and generates forward using temperature-scaled sampling.
    """

    PERIOD_DEFLECTIONS = [
        "I am not well acquainted with such novelties, but I am assured that",
        "Indeed, such matters are little spoken of in Hertfordshire, though",
        "Upon my word, I cannot pretend to understand such things, yet",
        "I have never considered such an affair, but I am convinced that",
        "It is a truth universally acknowledged that",
    ]

    def extract_keywords(self, text: str) -> List[Tuple[str, ...]]:
        """Extract multi-token phrases and single keywords present in corpus."""
        tokens = preprocess_text(text)
        if not tokens:
            return []

        # Filter out stopwords to find core content words
        content_tokens = [tok for tok in tokens if tok not in DEFAULT_STOPWORDS]

        candidates = []

        # 1. Check for 2-token phrases where at least one word is a content word
        # (Allows titles like 'mr darcy', 'miss bingley', etc.)
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if (t1 in content_tokens or t1 in ("mr", "mrs", "miss", "lady")) and (
                t2 in content_tokens
            ):
                pair = (t1, t2)
                if pair in self.bigram_counts:
                    candidates.append(pair)

        # 2. Search for meaningful single content words in vocabulary
        for tok in content_tokens:
            if tok in self.vocabulary:
                candidates.append((tok,))

        return candidates

    def respond(self, user_input: str) -> str:
        """Generate a response conditioned on user input keywords."""
        keywords = self.extract_keywords(user_input)

        # Case A: Out-of-vocabulary / No known keywords
        if not keywords:
            opener = self.rng.choice(self.PERIOD_DEFLECTIONS)
            # Sample forward from <s> to complete the thought
            continuation = self._generate_from_context(
                context=(BOS,), max_tokens=self.max_len - 10
            )
            return self.format_response(opener.split() + continuation)

        # Case B: Multi-token phrase found (e.g. ('mr', 'darcy'))
        target_phrase = keywords[0]

        if len(target_phrase) == 2:
            w1, w2 = target_phrase
            continuation = self._generate_from_context(
                context=(w1, w2), max_tokens=self.max_len
            )
            return self.format_response([w1, w2] + continuation)

        # Case C: Single keyword found (e.g. ('marriage',))
        kw = target_phrase[0]

        # Find bigrams starting with keyword
        matching_bigrams = [
            (b[1], count)
            for b, count in self.bigram_counts.items()
            if b[0] == kw and b[1] not in (BOS, EOS)
        ]

        if matching_bigrams:
            words, counts = zip(*matching_bigrams)
            next_word = sample_with_temperature(
                words, counts, temperature=self.temperature, rng=self.rng
            )
            continuation = self._generate_from_context(
                context=(kw, next_word), max_tokens=self.max_len
            )
            return self.format_response([kw, next_word] + continuation)
        else:
            # Fallback: start from BOS and generate
            continuation = self._generate_from_context(
                context=(BOS,), max_tokens=self.max_len
            )
            return self.format_response([kw] + continuation)

    def _generate_from_context(
        self, context: Tuple[str, ...], max_tokens: int = 30
    ) -> List[str]:
        """Generate a token sequence starting from an initial context."""
        generated: List[str] = []
        curr_context = context

        for _ in range(max_tokens):
            if self.n == 3 and len(curr_context) >= 2:
                pair = (curr_context[-2], curr_context[-1])
                followers = [
                    (t[2], c)
                    for t, c in self.trigram_counts.items()
                    if t[:2] == pair and t[2] != BOS
                ]
                if not followers:
                    # Backoff to bigram
                    followers = [
                        (b[1], c)
                        for b, c in self.bigram_counts.items()
                        if b[0] == curr_context[-1] and b[1] != BOS
                    ]
            else:
                last_word = curr_context[-1]
                followers = [
                    (b[1], c)
                    for b, c in self.bigram_counts.items()
                    if b[0] == last_word and b[1] != BOS
                ]

            if not followers:
                break

            # Avoid premature termination when response is too short
            if len(generated) < 3 and any(f[0] != EOS for f in followers):
                non_eos_followers = [f for f in followers if f[0] != EOS]
                if non_eos_followers:
                    followers = non_eos_followers

            words, counts = zip(*followers)
            next_token = sample_with_temperature(
                words, counts, temperature=self.temperature, rng=self.rng
            )

            if next_token == EOS:
                break

            generated.append(next_token)
            curr_context = curr_context + (next_token,)

        return generated


# Registry mapping string identifiers to chatbot classes
CHATBOT_REGISTRY: Dict[str, type] = {
    "keyword": KeywordSeededChatbot,
}
