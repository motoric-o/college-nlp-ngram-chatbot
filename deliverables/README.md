# Course & Project Deliverables: N-Gram Language Modeling & Chatbot Exploration

This directory contains the complete deliverables for the Natural Language Processing (NLP) course assignment along with the multi-phase conversational chatbot architectures developed as an exploration.

Each model has been organized into its own dedicated subfolder containing its focused **Jupyter Notebook (`.ipynb`)**, its **5-sentence generation script**, and its **exported text deliverable**.

---

## Directory Overview

```text
deliverables/
├── README.md                                  # This directory guide
├── generate_all_model_sentences.py            # Master runner: generates 5 sentences for all models
├── all_generated_sentences.txt                # Consolidated deliverable text for all models
├── build_notebooks.py                         # Programmatic generator for all 5 notebooks
│
├── 01_core_ngram_models/                      # [Core Assignment: Tasks 1 to 17]
│   ├── Core_Ngram_Language_Models.ipynb       # Complete notebook (Tasks 1-17, MLE, Laplace, Perplexity)
│   ├── generate_5_sentences.py                # Generates 5 sentences for Bigram, Trigram, Unigram, Laplace
│   └── generated_sentences.txt                # 5-sentence deliverable text file
│
├── 02_keyword_seeded_chatbot/                 # [Phase 1 Chatbot Exploration]
│   ├── Keyword_Seeded_Chatbot.ipynb           # Temperature-scaled keyword anchoring notebook
│   ├── generate_5_sentences.py                # Generates 5 conversational turns on Pride & Prejudice
│   └── generated_sentences.txt                # 5-sentence conversational responses
│
├── 03_intelligent_austen_chatbot/             # [Phase 2 Victorian Persona Chatbot]
│   ├── Intelligent_Austen_Chatbot.ipynb       # Intent rules, character memory ('he'->Darcy), Best-of-K
│   ├── generate_5_sentences.py                # Generates 5 intelligent conversational turns
│   └── generated_sentences.txt                # 5-sentence conversational responses
│
├── 04_dailydialog_chatbot/                    # [Phase 2 Modern Daily Conversation Chatbot]
│   ├── DailyDialog_Casual_Chatbot.ipynb       # 140k-utterance DailyDialog adaptation notebook
│   ├── generate_5_sentences.py                # Generates 5 modern casual dialogue responses
│   └── generated_sentences.txt                # 5-sentence conversational responses
│
└── 05_smart_hybrid_chatbot/                   # [Phase 3 Retrieval-Augmented N-Gram Engine]
    ├── Smart_Hybrid_Chatbot.ipynb             # Inverted index + N-gram generative completion notebook
    ├── generate_5_sentences.py                # Generates 5 smart hybrid responses (shopping, pricing, etc.)
    └── generated_sentences.txt                # 5-sentence conversational responses
```

---

## Deliverables Summary

### 1. Core Assignment Deliverable (`01_core_ngram_models/`)
- **Fulfills**: Tasks 1 to 17 of assignment specifications.
- **Jupyter Notebook**: [`Core_Ngram_Language_Models.ipynb`](file:///d:/Personal%20Documents/Studies/College/Semester%205/NLP/N-Gram%20LM/ngram_chatbot/deliverables/01_core_ngram_models/Core_Ngram_Language_Models.ipynb)
- **5 Generated Sentences**: Required by Deliverable 3 in PDF Section 25.
  - Bigram Model (5 sentences)
  - Trigram Model (5 sentences)
  - Unigram Baseline (5 sentences)
  - Laplace-Smoothed Bigram Model (5 sentences)
- **Quick Run**:
  ```powershell
  python deliverables/01_core_ngram_models/generate_5_sentences.py
  ```

### 2. Keyword-Seeded Chatbot (`02_keyword_seeded_chatbot/`)
- **Fulfills**: Phase 1 conversational exploration.
- **Notebook**: [`Keyword_Seeded_Chatbot.ipynb`](file:///d:/Personal%20Documents/Studies/College/Semester%205/NLP/N-Gram%20LM/ngram_chatbot/deliverables/02_keyword_seeded_chatbot/Keyword_Seeded_Chatbot.ipynb)
- **Quick Run**:
  ```powershell
  python deliverables/02_keyword_seeded_chatbot/generate_5_sentences.py
  ```

### 3. Intelligent Austen Chatbot (`03_intelligent_austen_chatbot/`)
- **Fulfills**: Phase 2 rule-based intents, coreference resolution, and Best-of-K re-ranking.
- **Notebook**: [`Intelligent_Austen_Chatbot.ipynb`](file:///d:/Personal%20Documents/Studies/College/Semester%205/NLP/N-Gram%20LM/ngram_chatbot/deliverables/03_intelligent_austen_chatbot/Intelligent_Austen_Chatbot.ipynb)
- **Quick Run**:
  ```powershell
  python deliverables/03_intelligent_austen_chatbot/generate_5_sentences.py
  ```

### 4. DailyDialog Casual Chatbot (`04_dailydialog_chatbot/`)
- **Fulfills**: Phase 2 modern conversational corpus adaptation.
- **Notebook**: [`DailyDialog_Casual_Chatbot.ipynb`](file:///d:/Personal%20Documents/Studies/College/Semester%205/NLP/N-Gram%20LM/ngram_chatbot/deliverables/04_dailydialog_chatbot/DailyDialog_Casual_Chatbot.ipynb)
- **Quick Run**:
  ```powershell
  python deliverables/04_dailydialog_chatbot/generate_5_sentences.py
  ```

### 5. Smart Hybrid Chatbot (`05_smart_hybrid_chatbot/`)
- **Fulfills**: Phase 3 Retrieval-Augmented N-Gram architecture combining 76,052 dialogue pairs with generative N-gram completion.
- **Notebook**: [`Smart_Hybrid_Chatbot.ipynb`](file:///d:/Personal%20Documents/Studies/College/Semester%205/NLP/N-Gram%20LM/ngram_chatbot/deliverables/05_smart_hybrid_chatbot/Smart_Hybrid_Chatbot.ipynb)
- **Quick Run**:
  ```powershell
  python deliverables/05_smart_hybrid_chatbot/generate_5_sentences.py
  ```

---

## Master Generation Command
To run all 5 models and update all generated sentence files in one command:
```powershell
python deliverables/generate_all_model_sentences.py
```
This script updates each individual folder's `generated_sentences.txt` and compiles them into `deliverables/all_generated_sentences.txt`.
