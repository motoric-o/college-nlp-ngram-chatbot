"""N-Gram Chatbot Architecture and Implementations.

Provides:
  - BaseChatbot: Foundation class managing corpus data, n-gram tables, and temperature settings.
  - KeywordSeededChatbot: Conversational chatbot that extracts topic keywords from user messages
    and generates conditioned responses using temperature-scaled N-gram sampling.
  - CHATBOT_REGISTRY: Factory dictionary to easily register and instantiate different chatbot classes.
"""

import heapq
import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data import load_corpus, load_dialogue_pairs, set_seed
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

    PROPER_NOUNS: Dict[str, str] = AUSTEN_PROPER_NOUNS

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
            if tok in self.PROPER_NOUNS:
                formatted.append(self.PROPER_NOUNS[tok])
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


class IntelligentChatbot(KeywordSeededChatbot):
    """Chatbot implementing Phase 2: classical "intelligence" on top of N-grams.

    Pillars:
      1. Rule sets / intent classification (greeting, identity, farewell, opinion, thanks).
      2. Conversational memory & pronoun coreference resolution.
      3. Best-of-K candidate generation with multi-criteria re-ranking.
      4. ELIZA-style pronoun reflection and persona framing.
    """

    INTENT_PATTERNS = [
        ("GREETING", re.compile(r"^\s*(hello|hi|hey|greetings|good (morning|afternoon|evening|day)|how do you do)\b", re.I)),
        ("FAREWELL", re.compile(r"\b(goodbye|good bye|farewell|good night|adieu|bye)\b", re.I)),
        ("IDENTITY", re.compile(r"\b(who are you|what is your name|what's your name|are you a (bot|robot|machine))\b", re.I)),
        ("THANKS", re.compile(r"\b(thank you|thanks|much obliged)\b", re.I)),
        ("OPINION", re.compile(r"\b(?:what do you think (?:of|about)|do you like|how do you feel about|your opinion (?:of|on))\s+(.+)", re.I)),
    ]

    INTENT_RESPONSES = {
        "GREETING": [
            "Good day to you! I trust you find yourself in tolerable health.",
            "How do you do? It is a pleasure to make your acquaintance.",
            "Good day! Pray, what news do you bring from the neighbourhood?",
        ],
        "FAREWELL": [
            "I take my leave of you with the utmost civility. Farewell!",
            "Adieu! I hope we shall meet again before long.",
        ],
        "IDENTITY": [
            "I am but a humble acquaintance from Hertfordshire, fashioned from the pages of Miss Austen's work.",
            "I am a creature of words and probabilities, though I flatter myself my manners are tolerable.",
        ],
        "THANKS": [
            "You are very welcome; it is no trouble at all.",
            "Pray, do not mention it. I am happy to oblige.",
        ],
    }

    OPINION_FRAMES = [
        "Upon the subject of {topic}, I am inclined to think that",
        "As for {topic}, I must confess that",
        "Regarding {topic}, I daresay",
    ]

    PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "it"}

    REFLECTIONS = {
        "i": "you", "me": "you", "my": "your", "mine": "yours", "am": "are",
        "myself": "yourself", "you": "I", "your": "my", "yours": "mine",
        "yourself": "myself", "are": "am",
    }

    DANGLING = {
        "and", "or", "but", "to", "in", "of", "the", "a", "an", "with", "that",
        "which", "for", "on", "at", "by", "from", "as", "my", "your", "his",
        "her", "their", "mr", "mrs", "miss", "lady", "is", "was", "be", ",",
    }

    def __init__(self, *args, k_candidates: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_candidates = k_candidates
        self.history: List[Tuple[str, str]] = []
        self.last_entity: Optional[str] = None
        self.last_intent: Optional[str] = None
        self._build_indexes()

    # ------------------------------------------------------------------ indexes
    def _build_indexes(self):
        """Precompute follower tables so Best-of-K generation stays fast."""
        self.bi_followers: Dict[str, List[Tuple[str, int]]] = {}
        for (w1, w2), c in self.bigram_counts.items():
            if w2 != BOS:
                self.bi_followers.setdefault(w1, []).append((w2, c))
        self.tri_followers: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
        for (w1, w2, w3), c in self.trigram_counts.items():
            if w3 != BOS:
                self.tri_followers.setdefault((w1, w2), []).append((w3, c))
        self.ctx_totals_bi = {w: sum(c for _, c in f) for w, f in self.bi_followers.items()}
        self.ctx_totals_tri = {p: sum(c for _, c in f) for p, f in self.tri_followers.items()}
        # Known entities = proper nouns present in the vocabulary
        self.entities = {w for w in self.PROPER_NOUNS if w in self.vocabulary
                         and w not in ("mr", "mrs", "miss", "lady")}

    def _generate_from_context(self, context: Tuple[str, ...], max_tokens: int = 30) -> List[str]:
        generated: List[str] = []
        ctx = context
        hit_eos = False
        for _ in range(max_tokens):
            followers = None
            if self.n == 3 and len(ctx) >= 2:
                followers = self.tri_followers.get((ctx[-2], ctx[-1]))
            if not followers:
                followers = self.bi_followers.get(ctx[-1])
            if not followers:
                break
            if len(generated) < 3:
                non_eos = [f for f in followers if f[0] != EOS]
                if non_eos:
                    followers = non_eos
            words, counts = zip(*followers)
            tok = sample_with_temperature(words, counts, temperature=self.temperature, rng=self.rng)
            if tok == EOS:
                hit_eos = True
                break
            generated.append(tok)
            ctx = ctx + (tok,)
        self._last_hit_eos = hit_eos
        return generated

    # ------------------------------------------------------------------ pillar 1
    def classify_intent(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (intent, captured_topic) using ordered regex rules."""
        for name, pat in self.INTENT_PATTERNS:
            m = pat.search(text)
            if m:
                topic = m.group(m.lastindex) if name == "OPINION" and m.lastindex else None
                return name, topic
        return None, None

    # ------------------------------------------------------------------ pillar 2
    def resolve_coreference(self, tokens: List[str]) -> List[str]:
        """Replace 3rd-person pronouns with the remembered entity when no entity is named."""
        if any(t in self.entities for t in tokens) or not self.last_entity:
            return tokens
        return [self.last_entity if t in self.PRONOUNS else t for t in tokens]

    def _update_memory(self, tokens: List[str]):
        named = [t for t in tokens if t in self.entities]
        if named:
            self.last_entity = named[-1]

    # ------------------------------------------------------------------ pillar 4
    def reflect(self, text: str) -> str:
        """ELIZA-style pronoun reversal (first <-> second person)."""
        return " ".join(self.REFLECTIONS.get(w.lower(), w) for w in text.split())

    # ------------------------------------------------------------------ pillar 3
    def _log_prob(self, seq: List[str]) -> float:
        """Average add-1 smoothed log-probability (with bigram backoff) of a sequence."""
        V = len(self.vocabulary) + 1
        total, n = 0.0, 0
        for i in range(1, len(seq)):
            w = seq[i]
            if self.n == 3 and i >= 2 and (seq[i - 2], seq[i - 1]) in self.ctx_totals_tri:
                c = self.trigram_counts.get((seq[i - 2], seq[i - 1], w), 0)
                d = self.ctx_totals_tri[(seq[i - 2], seq[i - 1])]
            else:
                c = self.bigram_counts.get((seq[i - 1], w), 0)
                d = self.ctx_totals_bi.get(seq[i - 1], 0)
            total += math.log((c + 1) / (d + V))
            n += 1
        return total / max(n, 1)

    def score_candidate(self, tokens: List[str], topics: set, ended: bool) -> float:
        if not tokens:
            return -1e9
        relevance = 2.0 * len(set(tokens) & topics)
        fluency = self._log_prob([BOS] + tokens + ([EOS] if ended else []))
        dangling = 5.0 if tokens[-1] in self.DANGLING else 0.0
        length = 0.0
        if len(tokens) < 4:
            length += 3.0
        if len(tokens) > 25:
            length += 0.2 * (len(tokens) - 25)
        if not ended:
            length += 1.5
        grams = list(zip(tokens, tokens[1:]))
        repetition = 1.5 * (len(grams) - len(set(grams)))
        return relevance + fluency - dangling - length - repetition

    def _best_of_k(self, seed: List[str], topics: set) -> List[str]:
        best, best_score = seed, -1e18
        for _ in range(self.k_candidates):
            cont = self._generate_from_context(tuple([BOS] + seed), max_tokens=self.max_len)
            cand = seed + cont
            s = self.score_candidate(cand, topics, self._last_hit_eos)
            if s > best_score:
                best, best_score = cand, s
        self.last_scores = best_score
        return best

    def _seed_for(self, tokens: List[str]) -> List[str]:
        """Pick a seed (1-2 tokens) from the user's content words."""
        kws = self.extract_keywords(" ".join(tokens))
        if not kws:
            return []
        # Prefer entity-bearing phrases
        kws.sort(key=lambda k: (not any(t in self.entities for t in k), -len(k)))
        return list(kws[0])

    # ------------------------------------------------------------------ pipeline
    def respond(self, user_input: str) -> str:
        intent, topic = self.classify_intent(user_input)
        self.last_intent = intent

        if intent in self.INTENT_RESPONSES:
            reply = self.rng.choice(self.INTENT_RESPONSES[intent])
            self.history.append((user_input, reply))
            return reply

        tokens = preprocess_text(topic if intent == "OPINION" else user_input)
        tokens = self.resolve_coreference(tokens)
        self._update_memory(tokens)
        topics = {t for t in tokens if t not in DEFAULT_STOPWORDS and t in self.vocabulary}

        seed = self._seed_for(tokens)
        is_oov = not seed
        if is_oov and self.last_entity:
            seed = [self.last_entity]  # fall back to conversational memory
        body = self._best_of_k(seed, topics)

        prefix: List[str] = []
        if intent == "OPINION":
            topic_text = " ".join(self.PROPER_NOUNS.get(t, t) for t in (seed or tokens))
            prefix = self.rng.choice(self.OPINION_FRAMES).format(topic=topic_text).split()
        elif is_oov:
            prefix = self.rng.choice(self.PERIOD_DEFLECTIONS).split()
        elif re.match(r"^\s*i\b", user_input, re.I) and self.rng.random() < 0.5:
            prefix = (f"You say {self.reflect(user_input.strip().rstrip('.!?'))}? Well,").split()

        reply = self.format_response(prefix + body)
        self.history.append((user_input, reply))
        return reply


class DailyDialogChatbot(IntelligentChatbot):
    """IntelligentChatbot adapted to modern casual English (DailyDialog corpus).

    Same 4 pillars, but with a modern persona and topic-based memory: DailyDialog has
    few recurring named characters, so 'it'/'that'/'they' resolve to the last topic noun.
    """

    PROPER_NOUNS: Dict[str, str] = {}

    INTENT_RESPONSES = {
        "GREETING": [
            "Hey! How's it going?",
            "Hi there! What's up?",
            "Hello! Nice to meet you. How are you today?",
        ],
        "FAREWELL": [
            "Bye! Take care.",
            "See you later! Have a nice day.",
        ],
        "IDENTITY": [
            "I'm a little chatbot built from a trigram model of everyday conversations.",
            "Just a statistical chatbot. I learned to talk from thousands of daily dialogues.",
        ],
        "THANKS": [
            "You're welcome!",
            "No problem at all.",
        ],
    }

    OPINION_FRAMES = [
        "Honestly, about {topic}, I think",
        "Well, when it comes to {topic},",
        "Hmm, {topic}? I guess",
    ]

    PERIOD_DEFLECTIONS = [
        "I'm not really sure about that, but",
        "Hmm, I don't know much about that. Anyway,",
        "That's a tough one. I guess",
    ]

    TOPIC_PRONOUNS = {"it", "that", "this", "they", "them"}

    def __init__(self, corpus_name: str = "dailydialog", *args, **kwargs):
        if corpus_name in ("pride", "pride_and_prejudice"):
            corpus_name = "dailydialog"  # this persona only makes sense on dialogue data
        super().__init__(corpus_name, *args, **kwargs)
        self.last_topic: Optional[str] = None

    def _topic_words(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in DEFAULT_STOPWORDS
                and t not in self.TOPIC_PRONOUNS and t in self.vocabulary and len(t) > 2]

    def resolve_coreference(self, tokens: List[str]) -> List[str]:
        if not self.last_topic:
            return tokens
        return [self.last_topic if t in self.TOPIC_PRONOUNS else t for t in tokens]

    def _update_memory(self, tokens: List[str]):
        words = self._topic_words(tokens)
        if self.last_topic in words:
            return  # still talking about the same thing
        if words:
            self.last_topic = words[-1]
            self.last_entity = self.last_topic  # keeps /history and fallback seeding working


DIALOGUE_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "of", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "it", "this", "that",
}


def extract_dialogue_features(text: str) -> set:
    """Extract unigrams and bigrams from dialogue text for fast inverted indexing."""
    tokens = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    unigrams = [w for w in tokens if w not in DIALOGUE_STOPWORDS]
    bigrams = [
        f"{tokens[i]}_{tokens[i+1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] not in DIALOGUE_STOPWORDS or tokens[i + 1] not in DIALOGUE_STOPWORDS
    ]
    return set(unigrams + bigrams)


class DialoguePairIndex:
    """Inverted index with BM25/TF-IDF scoring over dialogue prompt-response pairs."""

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs
        self.inv_index: Dict[str, List[int]] = defaultdict(list)
        for idx, (p, _) in enumerate(pairs):
            feats = extract_dialogue_features(p)
            for f in feats:
                self.inv_index[f].append(idx)
        self.N = len(pairs)
        self.idf = {
            f: math.log((self.N + 1) / (len(p) + 1)) + 1.0
            for f, p in self.inv_index.items()
        }

    def search(self, query: str, top_m: int = 5) -> List[Tuple[Tuple[str, str], float]]:
        feats = extract_dialogue_features(query)
        scores: Dict[int, float] = defaultdict(float)
        for f in feats:
            if f in self.inv_index:
                w_idf = self.idf[f]
                for idx in self.inv_index[f]:
                    scores[idx] += w_idf
        if not scores:
            return []
        top = heapq.nlargest(top_m, scores.items(), key=lambda x: x[1])
        return [(self.pairs[idx], score) for idx, score in top]


class SmartChatbot(DailyDialogChatbot):
    """Smart Hybrid Dialogue Chatbot.

    Combines:
      1. Conversational intent / speech-act handling.
      2. Dialogue turn-pair inverted index (DailyDialog) for authentic conversational responses.
      3. Temperature-scaled N-gram generative completion and hybrid synthesis.
      4. Coreference / pronoun memory tracking across turns.
      5. Multi-criteria re-ranking: dialogue relevance, topic overlap, and N-gram fluency.
    """

    _CACHED_INDEX: Optional[DialoguePairIndex] = None

    def __init__(self, corpus_name: str = "dailydialog", *args, **kwargs):
        super().__init__(corpus_name=corpus_name, *args, **kwargs)
        self.dialogue_index = self._get_or_build_index()

    def _get_or_build_index(self) -> DialoguePairIndex:
        if SmartChatbot._CACHED_INDEX is not None:
            return SmartChatbot._CACHED_INDEX

        if self.quick:
            sample_pairs = [
                ("Hello", "Hi there! How can I help you?"),
                ("I want to buy a computer", "Well, you can get an excellent deal on a new computer here."),
                ("How much does it cost?", "It costs 125 dollars a month."),
            ]
            SmartChatbot._CACHED_INDEX = DialoguePairIndex(sample_pairs)
        else:
            pairs = load_dialogue_pairs()
            SmartChatbot._CACHED_INDEX = DialoguePairIndex(pairs)

        return SmartChatbot._CACHED_INDEX

    def _format_raw_reply(self, text: str) -> str:
        """Clean spacing and capitalization for raw dialogue replies."""
        cleaned = re.sub(r"\s+([',.!?;:])", r"\1", text.strip())
        cleaned = re.sub(r"([A-Za-z])\s+('\w+)", r"\1\2", cleaned)
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            if cleaned[-1] not in ".!?":
                cleaned += "."
        return cleaned

    def respond(self, user_input: str) -> str:
        # 1. Intent classification
        intent, topic = self.classify_intent(user_input)
        self.last_intent = intent

        if intent in self.INTENT_RESPONSES:
            reply = self.rng.choice(self.INTENT_RESPONSES[intent])
            self.history.append((user_input, reply))
            return reply

        # 2. Extract content tokens & update conversational memory
        tokens = preprocess_text(topic if intent == "OPINION" else user_input)
        resolved_tokens = self.resolve_coreference(tokens)
        self._update_memory(resolved_tokens)

        content_words = {
            t for t in resolved_tokens
            if t not in DEFAULT_STOPWORDS and len(t) > 2
        }

        # Enrich query with last_topic if pronoun is used without a major noun
        enriched_query = user_input
        if any(p in tokens for p in self.TOPIC_PRONOUNS) and self.last_topic:
            enriched_query = f"{user_input} {self.last_topic}"

        # 3. Retrieve dialogue pairs
        retrieved = self.dialogue_index.search(enriched_query, top_m=5)

        candidates = []
        for (p, r), match_score in retrieved:
            r_tokens = preprocess_text(r)
            if r_tokens:
                # Channel A: Authentic dialogue response
                formatted_r = self._format_raw_reply(r)
                candidates.append((r_tokens, formatted_r, match_score, "retrieval"))

                # Channel B: N-Gram hybrid continuation seeded from response opener
                if len(r_tokens) >= 2:
                    opener = tuple(r_tokens[:2])
                    cont = self._generate_from_context(tuple([BOS] + list(opener)), max_tokens=self.max_len)
                    if cont:
                        hybrid_tokens = list(opener) + cont
                        candidates.append((hybrid_tokens, self.format_response(hybrid_tokens), match_score * 0.7, "hybrid"))

        # Channel C: Pure N-Gram generation from topic seed or memory
        seed = self._seed_for(resolved_tokens)
        if not seed and self.last_topic:
            seed = [self.last_topic]
        if seed:
            cont = self._generate_from_context(tuple([BOS] + seed), max_tokens=self.max_len)
            pure_tokens = seed + cont
            candidates.append((pure_tokens, self.format_response(pure_tokens), 0.0, "pure_ngram"))

        if not candidates:
            # Fallback
            prefix = self.rng.choice(self.PERIOD_DEFLECTIONS).split()
            cont = self._generate_from_context((BOS,), max_tokens=self.max_len)
            reply = self.format_response(prefix + cont)
            self.history.append((user_input, reply))
            return reply

        # 4. Multi-Criteria Scoring: Fluency + Match + Topic Overlap - Penalties
        scored_candidates = []
        for cand_tokens, cand_text, match_score, c_type in candidates:
            lm_score = self._log_prob([BOS] + cand_tokens + [EOS])
            topic_overlap = len(set(cand_tokens) & content_words)
            dangling = 5.0 if cand_tokens[-1] in self.DANGLING else 0.0
            length_pen = 0.0
            if len(cand_tokens) < 3:
                length_pen += 3.0
            elif len(cand_tokens) > 25:
                length_pen += 0.1 * (len(cand_tokens) - 25)

            grams = list(zip(cand_tokens, cand_tokens[1:]))
            repetition = 1.5 * (len(grams) - len(set(grams)))

            total_score = (
                (match_score * 0.4)
                + (topic_overlap * 3.5)
                + lm_score
                - dangling
                - length_pen
                - repetition
            )
            scored_candidates.append((total_score, cand_text))

        scored_candidates.sort(key=lambda x: -x[0])
        best_reply = scored_candidates[0][1]

        # Final persona wrap for opinion intent if present
        if intent == "OPINION":
            topic_str = " ".join(content_words) if content_words else "that"
            opener = self.rng.choice(self.OPINION_FRAMES).format(topic=topic_str)
            best_reply = f"{opener} {best_reply[0].lower() + best_reply[1:]}"

        self.history.append((user_input, best_reply))
        return best_reply


# Registry mapping string identifiers to chatbot classes
CHATBOT_REGISTRY: Dict[str, type] = {
    "keyword": KeywordSeededChatbot,
    "intelligent": IntelligentChatbot,
    "daily": DailyDialogChatbot,
    "smart": SmartChatbot,
}
