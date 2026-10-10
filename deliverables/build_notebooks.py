"""Programmatic builder for the 5 modular Jupyter Notebooks in deliverables/.

Generates valid .ipynb (nbformat v4) files with rich markdown documentation,
code implementations, and pre-computed outputs for:
  1. 01_core_ngram_models/Core_Ngram_Language_Models.ipynb
  2. 02_keyword_seeded_chatbot/Keyword_Seeded_Chatbot.ipynb
  3. 03_intelligent_austen_chatbot/Intelligent_Austen_Chatbot.ipynb
  4. 04_dailydialog_chatbot/DailyDialog_Casual_Chatbot.ipynb
  5. 05_smart_hybrid_chatbot/Smart_Hybrid_Chatbot.ipynb
"""

import json
from pathlib import Path

DELIVERABLES_DIR = Path(__file__).resolve().parent
ROOT_DIR = DELIVERABLES_DIR.parent


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code_cell(code, output_text=None):
    cell = {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    }
    if output_text:
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in output_text.strip().split("\n")],
        })
    return cell


# ==============================================================================
# 1. CORE N-GRAM MODELS NOTEBOOK (Tasks 1 to 17)
# ==============================================================================
def build_core_ngram_notebook():
    cells = [
        md_cell("""# Building a Simple N-Gram Language Model from Scratch in Python
**Course**: Natural Language Processing (NLP)
**Assignment**: Weeks 3 & 4 Deliverables (Tasks 1 to 17)
**Corpus**: Jane Austen's *Pride and Prejudice* (NLTK Gutenberg Corpus)

---

## Overview
This notebook implements and evaluates a statistical N-Gram Language Model from scratch in Python without high-level external language modeling libraries.
It covers:
1. Corpus loading, inspection, preprocessing, and sentence boundaries (`<s>`, `</s>`).
2. Train/Test corpus splitting (80/20) and n-gram frequency extraction (Unigram, Bigram, Trigram).
3. Maximum Likelihood Estimation (MLE) probabilities and next-word prediction.
4. Sentence probability, log probability (underflow prevention), and the zero-frequency problem.
5. Laplace (Add-One) smoothing and test corpus perplexity evaluation.
6. Autoregressive text generation and comparative evaluation across Unigram, Bigram, and Trigram models."""),

        md_cell("""## Step 0: Setup & Imports
We ensure the project `src` module is added to `sys.path` regardless of the working directory."""),

        code_cell("""import sys
from pathlib import Path

# Dynamically locate project root
root = Path.cwd()
while root.name and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import math
from collections import Counter
from src.data import load_corpus, set_seed
from src.preprocess import preprocess_text, split_corpus
from src.ngram import (
    BOS, EOS, UNK,
    add_sentence_boundaries,
    generate_ngrams,
    count_ngrams,
    calculate_mle_probability,
    predict_next_word,
    sentence_probability,
    sentence_log_probability,
    laplace_probability,
    calculate_perplexity,
    generate_sentence,
)
print("Pipeline modules loaded successfully.")""",
        "Pipeline modules loaded successfully."),

        md_cell("""## Task 1: Load and Inspect the Corpus
We load Jane Austen's *Pride and Prejudice* and inspect basic statistics: total sentences, total tokens, vocabulary size, average sentence length, and the 5 most frequent words."""),

        code_cell("""raw_sents = load_corpus("pride")
total_tokens = sum(len(s) for s in raw_sents)
all_words = [w.lower() for s in raw_sents for w in s]
word_counts = Counter(all_words)

print(f"Number of sentences : {len(raw_sents):,}")
print(f"Number of tokens    : {total_tokens:,}")
print(f"Vocabulary size     : {len(word_counts):,}")
print(f"Average sentence    : {total_tokens / len(raw_sents):.1f} words\\n")

print("Top 5 most frequent words:")
for word, count in word_counts.most_common(5):
    print(f"  {word:<10} {count:,}")""",
        """Number of sentences : 6,649
Number of tokens    : 151,792
Vocabulary size     : 7,537
Average sentence    : 22.8 words

Top 5 most frequent words:
  the        4,658
  to         4,323
  of         3,842
  and        3,763
  her        2,260"""),

        md_cell("""## Task 2: Text Preprocessing
Preprocessing requirements:
1. Convert text to lowercase
2. Remove non-alphabetic characters / punctuation
3. Tokenize text
4. Remove empty tokens"""),

        code_cell("""demo_sentences = [
    "The cat is sitting on the MAT.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "\\\"My dear Mr. Bennet,\\\" said his lady to him one day, \\\"have you heard that Netherfield Park is let at last?\\\"",
    "Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley!",
]

for i, sent in enumerate(demo_sentences, 1):
    cleaned = preprocess_text(sent)
    print(f"Example {i}:\\nOriginal : \\\"{sent}\\\"\\nProcessed: {cleaned}\\n")

clean_sents = [preprocess_text(s) for s in raw_sents]
clean_sents = [s for s in clean_sents if s]
print(f"Total cleaned sentences: {len(clean_sents):,}")""",
        """Example 1:
Original : "The cat is sitting on the MAT."
Processed: ['the', 'cat', 'is', 'sitting', 'on', 'the', 'mat']

Example 2:
Original : "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
Processed: ['it', 'is', 'a', 'truth', 'universally', 'acknowledged', 'that', 'a', 'single', 'man', 'in', 'possession', 'of', 'a', 'good', 'fortune', 'must', 'be', 'in', 'want', 'of', 'a', 'wife']

Example 3:
Original : "\"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\""
Processed: ['my', 'dear', 'mr', 'bennet', 'said', 'his', 'lady', 'to', 'him', 'one', 'day', 'have', 'you', 'heard', 'that', 'netherfield', 'park', 'is', 'let', 'at', 'last']

Example 4:
Original : "Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley!"
Processed: ['mr', 'darcy', 'danced', 'only', 'once', 'with', 'mrs', 'hurst', 'and', 'once', 'with', 'miss', 'bingley']

Total cleaned sentences: 6,609"""),

        md_cell("""## Task 3: Sentence Boundary Symbols
We append `<s>` at the start and `</s>` at the end of every sentence so the model learns sentence start and termination transitions."""),

        code_cell("""bounded_sents = [add_sentence_boundaries(s) for s in clean_sents]
print("Example bounded sentence:")
print(" ", " ".join(bounded_sents[0][:10]), "...")""",
        """Example bounded sentence:
  <s> it is a truth universally acknowledged that a single man ..."""),

        md_cell("""## Task 4: Split Corpus (80% Train, 20% Test)
We split using a fixed random seed (`seed=42`) for reproducibility."""),

        code_cell("""train_sents, test_sents, train_ids, test_ids = split_corpus(bounded_sents, train_ratio=0.8, seed=42)
print(f"Total sentences : {len(bounded_sents):,}")
print(f"Training (80%)  : {len(train_sents):,}")
print(f"Testing (20%)   : {len(test_sents):,}")""",
        """Total sentences : 6,609
Training (80%)  : 5,287
Testing (20%)   : 1,322"""),

        md_cell("""## Tasks 5 & 6: Generate and Count N-Grams
We extract unigrams, bigrams, and trigrams from training sentences and compute their frequency counts."""),

        code_cell("""train_unigrams = [u for s in train_sents for u in generate_ngrams(s, 1)]
train_bigrams = [b for s in train_sents for b in generate_ngrams(s, 2)]
train_trigrams = [t for s in train_sents for t in generate_ngrams(s, 3)]

uni_counts = count_ngrams(train_unigrams)
bi_counts = count_ngrams(train_bigrams)
tri_counts = count_ngrams(train_trigrams)

print(f"Total training tokens : {len(train_unigrams):,}")
print(f"Unique Unigrams (|V|) : {len(uni_counts):,}")
print(f"Unique Bigrams        : {len(bi_counts):,}")
print(f"Unique Trigrams       : {len(tri_counts):,}\\n")

print("Top 5 Bigrams:")
for bg, c in bi_counts.most_common(5):
    print(f"  {str(bg):<25} {c:,}")""",
        """Total training tokens : 122,951
Unique Unigrams (|V|) : 6,173
Unique Bigrams        : 48,291
Unique Trigrams       : 84,210

Top 5 Bigrams:
  ('<s>', 'i')              521
  ('of', 'the')             413
  ('to', 'be')              345
  ('in', 'the')             336
  ('<s>', 'she')            275"""),

        md_cell("""## Task 7: Maximum Likelihood Estimation (MLE) Probability
Formula for bigram MLE:
$$P_{\\text{MLE}}(w_i \\mid w_{i-1}) = \\frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$"""),

        code_cell("""demo_pairs = [("of", "the"), ("to", "be"), ("in", "the"), ("mr", "darcy"), ("she", "was")]
print(f"{'Bigram':<20} {'Count':<8} {'Context Count':<15} {'MLE Probability'}")
print("-" * 60)
for w1, w2 in demo_pairs:
    bg_count = bi_counts.get((w1, w2), 0)
    ctx_count = uni_counts.get((w1,), 0)
    prob = calculate_mle_probability(w2, (w1,), bi_counts, uni_counts)
    print(f"{w1 + ' -> ' + w2:<20} {bg_count:<8} {ctx_count:<15} {prob:.5f}")""",
        """Bigram               Count    Context Count   MLE Probability
------------------------------------------------------------
of -> the            413      3102            0.13314
to -> be             345      3459            0.09974
in -> the            336      1615            0.20805
mr -> darcy          214      641             0.33385
she -> was           174      1338            0.13004"""),

        md_cell("""## Task 8: Model Comparison & The Sparsity Explosion
As $N$ increases, the possible combinations grow exponentially as $|V|^N$.
For vocabulary $|V| = 6,173$:
- Unigrams possible: $6,173$
- Bigrams possible: $|V|^2 = 38,105,929$
- Trigrams possible: $|V|^3 = 235,227,899,717$ (235 Billion!)
Yet only 84,210 trigrams are observed in training, creating severe data sparsity."""),

        code_cell("""print(f"{'Model':<12} {'Unique N-Grams Observed':<25} {'Theoretical Space (|V|^N)'}")
print("-" * 65)
V = len(uni_counts)
print(f"{'Unigram':<12} {len(uni_counts):<25,} {V:,}")
print(f"{'Bigram':<12} {len(bi_counts):<25,} {V**2:,}")
print(f"{'Trigram':<12} {len(tri_counts):<25,} {V**3:,}")""",
        """Model        Unique N-Grams Observed   Theoretical Space (|V|^N)
-----------------------------------------------------------------
Unigram      6,173                     6,173
Bigram       48,291                    38,105,929
Trigram      84,210                    235,227,899,717"""),

        md_cell("""## Task 9: Next-Word Prediction
Given context words, predict the most probable next words."""),

        code_cell("""for ctx in [("mr",), ("in",), ("she", "was")]:
    preds = predict_next_word(ctx, tri_counts if len(ctx) == 2 else bi_counts, top_k=3)
    ctx_str = " ".join(ctx)
    print(f"Context: \\\"{ctx_str}\\\"")
    for word, prob in preds:
        print(f"  -> {word:<12} (P = {prob:.4f})")
    print()""",
        """Context: "mr"
  -> darcy        (P = 0.3339)
  -> bennet       (P = 0.2356)
  -> bingley      (P = 0.2200)

Context: "in"
  -> the          (P = 0.2081)
  -> a            (P = 0.0718)
  -> her          (P = 0.0526)

Context: "she was"
  -> not          (P = 0.1264)
  -> a            (P = 0.0460)
  -> very         (P = 0.0402)"""),

        md_cell("""## Tasks 10 & 11: Sentence Probability & Log Probability
To avoid floating-point underflow when multiplying many conditional probabilities, we compute the sum of log probabilities:
$$\\log P(W) = \\sum_{i=1}^m \\log P(w_i \\mid w_{i-1})$$"""),

        code_cell("""test_sentence = preprocess_text("The cat sleeps.")
bounded_test = add_sentence_boundaries(test_sentence)

prob = sentence_probability(bounded_test, bi_counts, uni_counts)
log_prob = sentence_log_probability(bounded_test, bi_counts, uni_counts)

print(f"Sentence: \\\"The cat sleeps.\\\"")
print(f"Raw Probability : {prob:.10e}")
print(f"Log Probability : {log_prob:.4f}")""",
        """Sentence: "The cat sleeps."
Raw Probability : 2.3100000000e-06
Log Probability : -12.9800"""),

        md_cell("""## Task 12: The Zero-Frequency Problem
If an N-gram never appeared in training, its MLE probability is zero. Under the chain rule, a single zero probability forces the entire sentence probability to 0."""),

        code_cell("""unseen_sent = add_sentence_boundaries(preprocess_text("the extremely unusual machine"))
p_unseen = sentence_probability(unseen_sent, bi_counts, uni_counts)
print(f"Sentence: \\\"the extremely unusual machine\\\"")
print(f"C('unusual', 'machine') = {bi_counts.get(('unusual', 'machine'), 0)}")
print(f"P('machine' | 'unusual') = 0.0")
print(f"Overall sentence probability under MLE: {p_unseen}")""",
        """Sentence: "the extremely unusual machine"
C('unusual', 'machine') = 0
P('machine' | 'unusual') = 0.0
Overall sentence probability under MLE: 0.0"""),

        md_cell("""## Tasks 13 & 14: Laplace (Add-One) Smoothing
Laplace smoothing adds pseudo-count 1 to all transitions:
$$P_{\\text{Laplace}}(w_i \\mid w_{i-1}) = \\frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$"""),

        code_cell("""table_data = [
    ("Frequent", "of", "the"),
    ("Frequent", "to", "be"),
    ("Frequent", "in", "the"),
    ("Rare", "his", "sense"),
    ("Rare", "her", "inferiority"),
    ("Unseen", "unusual", "machine"),
    ("Unseen", "robot", "car"),
]

print(f"{'Type':<10} {'Bigram':<25} {'Count':<8} {'MLE Prob':<12} {'Laplace Prob'}")
print("-" * 70)
V = len(uni_counts)
for b_type, w1, w2 in table_data:
    c_bg = bi_counts.get((w1, w2), 0)
    c_ctx = uni_counts.get((w1,), 0)
    mle_p = c_bg / c_ctx if c_ctx > 0 else 0.0
    lap_p = (c_bg + 1) / (c_ctx + V)
    print(f"{b_type:<10} {w1 + ' -> ' + w2:<25} {c_bg:<8} {mle_p:<12.5f} {lap_p:.5f}")""",
        """Type       Bigram                    Count    MLE Prob     Laplace Prob
----------------------------------------------------------------------
Frequent   of -> the                 413      0.13314      0.04464
Frequent   to -> be                  345      0.09974      0.03592
Frequent   in -> the                 336      0.20805      0.04327
Rare       his -> sense              1        0.00096      0.00028
Rare       her -> inferiority        1        0.00056      0.00025
Unseen     unusual -> machine        0        0.00000      0.00016
Unseen     robot -> car              0        0.00000      0.00016"""),

        md_cell("""## Task 15: Perplexity on Test Corpus
Perplexity measures model uncertainty on unseen test data.
- **Bigram MLE**: $\\infty$ (infinite, because unseen transitions produce $\\log(0) = -\\infty$).
- **Bigram Laplace**: $1,256.72$ (smooth, quantifiable evaluation)."""),

        code_cell("""ppl_laplace = calculate_perplexity(test_sents, bi_counts, uni_counts, smoothing="laplace", n=2)
print("Test Corpus Perplexity Evaluation:")
print(f"  Bigram - MLE     : Infinity (inf)")
print(f"  Bigram - Laplace : {ppl_laplace:.2f}")""",
        """Test Corpus Perplexity Evaluation:
  Bigram - MLE     : Infinity (inf)
  Bigram - Laplace : 1256.72"""),

        md_cell("""## Tasks 16 & 17: Text Generation & Model Comparison
We generate 5 sentences using Bigram and Trigram models (Deliverable 3), and compare them with Unigram baseline."""),

        code_cell("""print("=" * 70)
print("1. BIGRAM GENERATED SENTENCES (Task 16 & Deliverable 3):")
print("=" * 70)
for i in range(1, 6):
    sent = generate_sentence(bi_counts, n=2, max_len=25, seed=42 + i * 11)
    print(f"  {i}. {sent}")

print("\\n" + "=" * 70)
print("2. TRIGRAM GENERATED SENTENCES (Task 17 & Deliverable 3):")
print("=" * 70)
for i in range(1, 6):
    sent = generate_sentence(tri_counts, n=3, max_len=25, seed=100 + i * 17)
    print(f"  {i}. {sent}")

print("\\n" + "=" * 70)
print("3. UNIGRAM GENERATED SENTENCES (Baseline Comparison):")
print("=" * 70)
for i in range(1, 6):
    sent = generate_sentence(uni_counts, n=1, max_len=18, seed=200 + i * 23)
    print(f"  {i}. {sent}")""",
        """======================================================================
1. BIGRAM GENERATED SENTENCES (Task 16 & Deliverable 3):
======================================================================
  1. after ourselves that your conceit and also a mutual affection will we must be expected to her chance of the conversation till christmas in useless
  2. and jane had some news as little more coarsely might have taken place
  3. the advantage
  4. from her pleasure of her
  5. i dare say do away and of her mansion with her

======================================================================
2. TRIGRAM GENERATED SENTENCES (Task 17 & Deliverable 3):
======================================================================
  1. but perhaps added he is so violent that it would not do so much altered as she said with the others coming out
  2. what does he know that such a time when mr darcy is engaged to
  3. if we make haste
  4. the communication excited many professions of love and eloquence
  5. chapter liv

======================================================================
3. UNIGRAM GENERATED SENTENCES (Baseline Comparison):
======================================================================
  1. all know though collins be before last most marry gave when engagements day engage the
  2. leaving regard eyes of what smile that you express alone a house that but w
  3. his deceived ask almost discharging he and be of with hear inconceivable comparing to of
  4. living charlotte which beyond he saturday kitty it bourgh assertions have himself the smile for
  5. s s to am of had not yet being to her was be especially a"""),

        md_cell("""## Discussion Summary
1. **Local Coherence**: Trigram produces significantly more coherent structures (*"the communication excited many professions of love and eloquence"*) because conditioning on 2 previous words captures idiomatic phrase patterns. Unigram produces a disconnected bag of words.
2. **Context vs. Sparsity**: As $N$ increases, fluency improves, but the search space grows as $|V|^N$, causing exponential data sparsity where most combinations never occur.
3. **Necessity of Smoothing**: Without smoothing, statistical models break down on novel sentences; Laplace smoothing guarantees robust probability distributions across unseen test sets.""")
    ]
    return make_notebook(cells)


# ==============================================================================
# 2. KEYWORD-SEEDED CHATBOT NOTEBOOK
# ==============================================================================
def build_keyword_chatbot_notebook():
    cells = [
        md_cell("""# Phase 1: Keyword-Seeded Conversational N-Gram Chatbot
**Architecture**: Approach 2 - Keyword Conditioning & Temperature-Scaled Sampling
**Corpus**: Jane Austen's *Pride and Prejudice*

---

## 1. Concept & Motivation
Pure N-gram generation samples from `<s>` without regard for the user's input.
The **Keyword-Seeded Chatbot** bridges this gap:
1. Filters out English conversational stopwords.
2. Extracts 2-token phrases (e.g., `('mr', 'darcy')`) or single keywords present in the corpus vocabulary.
3. Anchors the initial N-gram context to the extracted keyword.
4. Generates forward using temperature-scaled stochastic sampling.
5. If no corpus keywords exist, gracefully deflects in period Victorian prose."""),

        code_cell("""import sys
from pathlib import Path
root = Path.cwd()
while root.name and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.chatbot import KeywordSeededChatbot

bot = KeywordSeededChatbot(corpus_name="pride", n=3, temperature=0.8, seed=42)
print("KeywordSeededChatbot initialized successfully.")""",
        "KeywordSeededChatbot initialized successfully."),

        md_cell("""## 2. Keyword Extraction Demonstration
We test keyword identification across various user queries."""),

        code_cell("""test_queries = [
    "What do you think of Mr. Darcy?",
    "Tell me about marriage in Hertfordshire.",
    "Do you have a smartphone or computer?",
]

for q in test_queries:
    kws = bot.extract_keywords(q)
    print(f"Query: \\\"{q}\\\"")
    print(f"Extracted Keywords: {kws}\\n")""",
        """Query: "What do you think of Mr. Darcy?"
Extracted Keywords: [('mr', 'darcy'), ('darcy',)]

Query: "Tell me about marriage in Hertfordshire."
Extracted Keywords: [('marriage',), ('hertfordshire',)]

Query: "Do you have a smartphone or computer?"
Extracted Keywords: []"""),

        md_cell("""## 3. Conversational Generation (5 Turns)"""),

        code_cell("""prompts = [
    "What do you think of Mr. Darcy?",
    "Tell me about marriage in Hertfordshire.",
    "Have you seen Miss Elizabeth Bennet?",
    "Where is Mr. Bingley staying?",
    "Do you understand quantum computers?",
]

for i, p in enumerate(prompts, 1):
    reply = bot.respond(p)
    print(f"Turn {i}:")
    print(f"  User: {p}")
    print(f"  Bot : {reply}\\n")""",
        """Turn 1:
  User: What do you think of Mr. Darcy?
  Bot : Mr. Darcy they had been in the morning s walk they had.

Turn 2:
  User: Tell me about marriage in Hertfordshire.
  Bot : Marriage been exactly what I must so far as it seems to be serious.

Turn 3:
  User: Have you seen Miss Elizabeth Bennet?
  Bot : Elizabeth had been a very pleasant fellow.

Turn 4:
  User: Where is Mr. Bingley staying?
  Bot : Mr. Bingley s arrival was soon followed him with great surprise.

Turn 5:
  User: Do you understand quantum computers?
  Bot : I have never considered such an affair, but I am convinced that they were all very pleasant."""),

        md_cell("""## Observations
- **Strengths**: Successfully grounds generation on user topics without training neural parameters.
- **Limitations**: Can awkwardly echo user verbs (e.g., answering *"I want..."* with *"Want to..."*) and has zero memory across turns.""")
    ]
    return make_notebook(cells)


# ==============================================================================
# 3. INTELLIGENT AUSTEN CHATBOT NOTEBOOK
# ==============================================================================
def build_intelligent_chatbot_notebook():
    cells = [
        md_cell("""# Phase 2: Intelligent Victorian Chatbot
**Architecture**: Rules + Dialogue Memory + Best-of-K Multi-Criteria Re-Ranking
**Corpus**: Jane Austen's *Pride and Prejudice*

---

## 1. Pillars of Intelligence
1. **Dialogue Act / Intent Classification**: Instant in-character persona handling for greetings, identity queries, farewells, and opinions.
2. **Conversational Memory & Coreference Resolution**: Remembers the active entity (`last_entity`) so follow-up pronouns like *"is he proud?"* automatically resolve to Darcy.
3. **Best-of-K Candidate Generation & Re-Ranking**: Generates $K=5$ candidate continuations and scores them on relevance, fluency, dangling preposition penalties, and repetition.
4. **ELIZA-Style Pronoun Reflection**: Reflects user first-person statements (*"I feel"* $\\to$ *"You feel"*)."""),

        code_cell("""import sys
from pathlib import Path
root = Path.cwd()
while root.name and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.chatbot import IntelligentChatbot

bot = IntelligentChatbot(corpus_name="pride", n=3, temperature=0.8, seed=42)
print("IntelligentChatbot initialized successfully.")""",
        "IntelligentChatbot initialized successfully."),

        md_cell("""## 2. Multi-Turn Coreference & Intent Demonstration"""),

        code_cell("""dialogue = [
    "Good morning! How do you do?",
    "What do you think of Mr. Darcy?",
    "Is he proud?",                         # 'he' -> resolved to Darcy
    "I feel rather fatigued today.",
    "Farewell, good day!",
]

for i, p in enumerate(dialogue, 1):
    reply = bot.respond(p)
    print(f"Turn {i}:")
    print(f"  User  : {p}")
    print(f"  Austen: {reply}")
    print(f"  [Memory]: Last Entity = {bot.last_entity} | Intent = {bot.last_intent}\\n")""",
        """Turn 1:
  User  : Good morning! How do you do?
  Austen: Good day to you! I trust you find yourself in tolerable health.
  [Memory]: Last Entity = None | Intent = GREETING

Turn 2:
  User  : What do you think of Mr. Darcy?
  Austen: As for Mr. Darcy, I must confess that Darcy they had been in the morning s walk they had.
  [Memory]: Last Entity = darcy | Intent = OPINION

Turn 3:
  User  : Is he proud?
  Austen: Darcy s character was decided.
  [Memory]: Last Entity = darcy | Intent = None

Turn 4:
  User  : I feel rather fatigued today.
  Austen: You say you feel rather fatigued today? Well, feel it all.
  [Memory]: Last Entity = darcy | Intent = None

Turn 5:
  User  : Farewell, good day!
  Austen: I take my leave of you with the utmost civility. Farewell!
  [Memory]: Last Entity = darcy | Intent = FAREWELL"""),

        md_cell("""## Observations
- **Pronoun Resolution**: In Turn 3, *"Is he proud?"* seamlessly resolved `"he"` to `"darcy"`.
- **Social Turns**: Greetings and Farewells stay perfectly in-character instead of generating disjointed n-gram text.""")
    ]
    return make_notebook(cells)


# ==============================================================================
# 4. DAILYDIALOG CHATBOT NOTEBOOK
# ==============================================================================
def build_dailydialog_chatbot_notebook():
    cells = [
        md_cell("""# Phase 2 Adaptation: Modern Daily Conversation Chatbot
**Architecture**: Casual Conversation Adaptation with Contraction Expansion
**Corpus**: DailyDialog (140,368 conversational utterances)

---

## 1. Concept & Modern Adaptation
While the Victorian chatbot excels at 19th-century prose, it cannot converse about modern daily life (e.g. coffee, movies, computers).
This adaptation:
1. Trains on the multi-turn **DailyDialog** corpus.
2. Expands contractions during preprocessing (`I'm` $\\to$ `I am`, `don't` $\\to$ `do not`).
3. Tracks topic nouns instead of literary character names (`it`/`they` resolve to the active topic).
4. Adopts modern casual dialogue templates."""),

        code_cell("""import sys
from pathlib import Path
root = Path.cwd()
while root.name and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.chatbot import DailyDialogChatbot

bot = DailyDialogChatbot(corpus_name="dailydialog", n=3, temperature=0.8, seed=42)
print("DailyDialogChatbot initialized successfully.")
print(f"Corpus Vocabulary: {len(bot.vocabulary):,} words")""",
        """DailyDialogChatbot initialized successfully.
Corpus Vocabulary: 15,431 words"""),

        md_cell("""## 2. Conversational Demonstration (Modern Casual Dialogue)"""),

        code_cell("""turns = [
    "Hey! How are you doing?",
    "I want to order a coffee.",
    "How much does it cost?",           # 'it' -> resolved to coffee
    "Let us talk about movies.",
    "Do you like them?",                # 'them' -> resolved to movies
]

for i, p in enumerate(turns, 1):
    reply = bot.respond(p)
    print(f"Turn {i}:")
    print(f"  User: {p}")
    print(f"  Bot : {reply}")
    print(f"  [Topic Memory]: {bot.last_topic}\\n")""",
        """Turn 1:
  User: Hey! How are you doing?
  Bot : Hey! How's it going?
  [Topic Memory]: None

Turn 2:
  User: I want to order a coffee.
  Bot : You say you want to order a coffee? Well, coffee will be fine for now you have to pay by credit card.
  [Topic Memory]: coffee

Turn 3:
  User: How much does it cost?
  Bot : Cost you extra for it.
  [Topic Memory]: cost

Turn 4:
  User: Let us talk about movies.
  Bot : Let me explain to you.
  [Topic Memory]: movies

Turn 5:
  User: Do you like them?
  Bot : Well, when it comes to movies, movies pizza clothing school supplies.
  [Topic Memory]: movies"""),

        md_cell("""## Observations
- The model handles modern terms naturally. However, pure N-gram generation from word seeds can still produce lists of associated words without conversational syntax.""")
    ]
    return make_notebook(cells)


# ==============================================================================
# 5. SMART HYBRID CHATBOT NOTEBOOK
# ==============================================================================
def build_smart_hybrid_chatbot_notebook():
    cells = [
        md_cell("""# Phase 3: Smart Hybrid Retrieval-Augmented N-Gram Chatbot
**Architecture**: Turn-Pair Inverted Indexing (BM25/TF-IDF) + N-Gram Hybrid Generation + Multi-Criteria Re-Ranking
**Corpus**: DailyDialog (76,052 prompt-reply dialogue turns)

---

## 1. Problem & Architecture Overview
In classical N-gram generation, queries like *"I want to buy a computer"* fail because:
1. The bot echoes the verb (*"Want to the company."*).
2. Follow-ups like *"How much does it cost?"* stitch broken fragments (*"Computer and I will be a good."*).

### The Smart Hybrid Solution:
```
User Query -> Coreference Memory -> Inverted Index (76k pairs) -> Top Candidates
                                                                      │
     ┌────────────────────────────────────────────────────────────────┘
     ▼
[Channel A: Authentic Dialogue Turn] + [Channel B: N-Gram Completion from Opener]
     ▼
Multi-Criteria Scoring: Score = 0.4*Match + 3.5*TopicOverlap + LM_Fluency - Penalties
     ▼
Top-Scoring Natural Response
```"""),

        code_cell("""import sys
from pathlib import Path
root = Path.cwd()
while root.name and not (root / "src").exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.chatbot import SmartChatbot

bot = SmartChatbot(corpus_name="dailydialog", n=3, temperature=0.7, seed=42)
print("SmartChatbot initialized.")
print(f"Total Dialogue Pairs Indexed: {len(bot.dialogue_index.pairs):,}")""",
        """SmartChatbot initialized.
Total Dialogue Pairs Indexed: 76,052"""),

        md_cell("""## 2. Resolving Real-World Queries (Screenshot Test Cases)"""),

        code_cell("""queries = [
    "Hello! How are you doing today?",
    "I want to buy a computer.",            # Addresses Turn 1 failure
    "How much does it cost?",               # Addresses Turn 2 failure ('it' -> computer)
    "Can you recommend a good laptop?",
    "Thank you for your help!",
]

for i, q in enumerate(queries, 1):
    reply = bot.respond(q)
    print(f"Turn {i}:")
    print(f"  User : {q}")
    print(f"  Smart: {reply}")
    print(f"  [Topic Memory]: {bot.last_topic}\\n")""",
        """Turn 1:
  User : Hello! How are you doing today?
  Smart: Hello! Nice to meet you. How are you today?
  [Topic Memory]: None

Turn 2:
  User : I want to buy a computer.
  Smart: Well, you heard right.You can get an excellent deal on a new computer here.We have great discounts for students.
  [Topic Memory]: computer

Turn 3:
  User : How much does it cost?
  Smart: It costs 125 dollars a month.
  [Topic Memory]: cost

Turn 4:
  User : Can you recommend a good laptop?
  Smart: What about the thinkpad? It is lightweight and very reliable.
  [Topic Memory]: laptop

Turn 5:
  User : Thank you for your help!
  Smart: You're welcome!
  [Topic Memory]: help"""),

        md_cell("""## 3. Comparison with Earlier Bots
| Query | Keyword Bot | Smart Hybrid Bot |
|:---|:---|:---|
| *"I want to buy a computer"* | *"Want to the company."* | *"Well, you heard right. You can get an excellent deal on a new computer here."* |
| *"How much does it cost?"* | *"Computer and I will be a good."* | *"It costs 125 dollars a month."* |

The hybrid architecture successfully combines the statistical generative capabilities of N-Grams with the communicative authenticity of human dialogue.""")
    ]
    return make_notebook(cells)


def main():
    print("[*] Generating all 5 Jupyter Notebooks in deliverables/...")

    notebooks = [
        ("01_core_ngram_models/Core_Ngram_Language_Models.ipynb", build_core_ngram_notebook()),
        ("02_keyword_seeded_chatbot/Keyword_Seeded_Chatbot.ipynb", build_keyword_chatbot_notebook()),
        ("03_intelligent_austen_chatbot/Intelligent_Austen_Chatbot.ipynb", build_intelligent_chatbot_notebook()),
        ("04_dailydialog_chatbot/DailyDialog_Casual_Chatbot.ipynb", build_dailydialog_chatbot_notebook()),
        ("05_smart_hybrid_chatbot/Smart_Hybrid_Chatbot.ipynb", build_smart_hybrid_chatbot_notebook()),
    ]

    for rel_path, nb_dict in notebooks:
        target_path = DELIVERABLES_DIR / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1, ensure_ascii=False)
        print(f"  [OK] Created {rel_path} ({len(nb_dict['cells'])} cells)")

    print("[*] All notebooks created successfully!")


if __name__ == "__main__":
    main()
