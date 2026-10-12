# N-Gram Language Model & Conversational Chatbot Exploration

An end-to-end implementation of statistical N-Gram Language Modeling from scratch in Python, spanning fundamental probabilistic modeling (Weeks 3 & 4 NLP course assignment) and multi-phase conversational chatbot architectures.

---

## 📌 Project Overview

This repository covers two major components:
1. **Assignment Tasks 1 to 17**: Core statistical N-Gram pipeline implemented from scratch without high-level NLP libraries (Corpus ingestion, text normalization, boundary markers, MLE estimation, Laplace smoothing, perplexity, and text generation).
2. **Conversational Chatbot Exploration**: Progressive architectural exploration using N-Gram models:
   - **Phase 1 (`keyword`)**: Keyword-seeded context anchoring with temperature-scaled sampling.
   - **Phase 2 (`intelligent`)**: Victorian Austen persona with dialogue act matching, conversational memory, coreference resolution, and Best-of-K re-ranking.
   - **Phase 2 Adaptation (`daily`)**: Modern casual English adaptation trained on 140,000+ utterances from DailyDialog.
   - **Phase 3 (`smart`)**: Hybrid retrieval-augmented N-Gram engine combining 76,052 dialogue turn pairs with generative N-gram synthesis.

---

## 📂 Repository Structure

```text
ngram_chatbot/
├── src/                                  # Core library package
│   ├── data.py                           # Corpus loaders (Pride & Prejudice, Brown, Reuters, DailyDialog)
│   ├── preprocess.py                     # Tokenization, lowercasing, vocabulary, and corpus splitting
│   ├── ngram.py                          # N-gram generation, counting, MLE, Laplace smoothing, PPL, sampling
│   ├── chatbot.py                        # Chatbot architectures (Keyword, Intelligent, Daily, Smart Hybrid)
│   └── eda.py                            # Corpus statistics and exploratory data analysis
│
├── tests/                                # Automated verification & assignment task runners
│   ├── tasks_1_to_6.py                   # Runner for Week 3 foundational pipeline tasks
│   ├── tasks_7_plus.py                   # Runner for Week 4 tasks (MLE, smoothing, perplexity, generation)
│   ├── test_intelligent_chatbot.py      # Test suite for Phase 2 Intelligent Victorian bot
│   └── test_smart_chatbot.py            # Test suite for Phase 3 Smart Hybrid bot
│
├── deliverables/                         # Course deliverables & model workspaces
│   ├── README.md                         # Detailed guide to all deliverables
│   ├── generate_all_model_sentences.py   # Master runner generating 5 sentences across all models
│   ├── all_generated_sentences.txt       # Consolidated export of all generated sentences
│   ├── build_notebooks.py                # Programmatic builder for all 5 Jupyter Notebooks
│   │
│   ├── 01_core_ngram_models(unused)/     # [Core Tasks 1 to 17]
│   │   ├── NgramLM_2472040_2472041_2472053_2472901.ipynb
│   │   ├── generate_5_sentences.py
│   │   └── generated_sentences.txt
│   │
│   ├── 02_keyword_seeded_chatbot/        # [Phase 1 Chatbot]
│   │   ├── NgramLM_Chatbot_KeywordSeeded_2472040_2472041_2472053_2472901.ipynb
│   │   ├── generate_5_sentences.py
│   │   └── generated_sentences.txt
│   │
│   ├── 03_intelligent_austen_chatbot/    # [Phase 2 Victorian Persona]
│   │   ├── NgramLM_Intelligent_2472040_2472041_2472053_2472901.ipynb
│   │   ├── generate_5_sentences.py
│   │   └── generated_sentences.txt
│   │
│   ├── 04_dailydialog_chatbot/           # [Phase 2 Modern Casual Adaptation]
│   │   ├── NgramLM_DailyDialogue_2472040_2472041_2472053_2472901.ipynb
│   │   ├── generate_5_sentences.py
│   │   └── generated_sentences.txt
│   │
│   └── 05_smart_hybrid_chatbot/          # [Phase 3 Retrieval-Augmented N-Gram]
│       ├── NgramLM_Hybrid_2472040_2472041_2472053_2472901.ipynb
│       ├── generate_5_sentences.py
│       └── generated_sentences.txt
│
└── main.py                               # Interactive terminal REPL for all chatbots
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment
Python 3.10+ is recommended. Set up and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # or pip install nltk matplotlib pandas ipykernel
```

### 2. Interactive Chatbot REPL (`main.py`)
Launch the terminal chat interface:

```powershell
# Default launch (runs Phase 3 Smart Hybrid Chatbot on DailyDialog)
python main.py

# Launch specific chatbot architectures
python main.py --bot smart        # Smart Hybrid Retrieval-Augmented N-Gram
python main.py --bot daily        # Modern Casual DailyDialog Chatbot
python main.py --bot intelligent  # Victorian Austen Persona Chatbot
python main.py --bot keyword      # Baseline Keyword-Seeded Chatbot
```

#### In-Chat Commands
- `/temp <val>` : Set sampling temperature (e.g., `/temp 0.5` for deterministic, `/temp 1.2` for creative).
- `/model <2|3>` : Switch between Bigram ($N=2$) and Trigram ($N=3$).
- `/bot <name>` : Switch chatbot class (`keyword`, `intelligent`, `daily`, `smart`) on the fly.
- `/history` : Display dialogue memory, remembered entity/topic, and recent conversation turns.
- `/info` : Show active model configuration, corpus, and vocabulary size.
- `exit` or `quit` : Exit the conversation.

---

## 🧪 Running Tests & Assignment Runners

All assignment tasks and chatbot capabilities are verified via automated scripts in `tests/`:

```powershell
# Run Assignment Tasks 1 to 6 (Foundation Pipeline)
python tests/tasks_1_to_6.py
python tests/tasks_1_to_6.py --quick     # Fast mode using sample sentences

# Run Assignment Tasks 7 to 17 (MLE, Smoothing, Perplexity, Generation)
python tests/tasks_7_plus.py
python tests/tasks_7_plus.py --quick

# Run Chatbot Test Suites
python tests/test_smart_chatbot.py
python tests/test_intelligent_chatbot.py
```

---

## 📦 Deliverables & 5 Generated Sentences

To generate the required **5 sentences** for Bigram, Trigram, and all developed models:

```powershell
# Master runner: Generates 5 sentences across all 5 model folders
python deliverables/generate_all_model_sentences.py
```

Individual model generation scripts:
```powershell
python "deliverables/01_core_ngram_models(unused)/generate_5_sentences.py"
python deliverables/02_keyword_seeded_chatbot/generate_5_sentences.py
python deliverables/03_intelligent_austen_chatbot/generate_5_sentences.py
python deliverables/04_dailydialog_chatbot/generate_5_sentences.py
python deliverables/05_smart_hybrid_chatbot/generate_5_sentences.py
```

All 5 modular Jupyter Notebooks can be re-generated or inspected at any time via:
```powershell
python deliverables/build_notebooks.py
```
