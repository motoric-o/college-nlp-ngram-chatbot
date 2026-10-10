"""Master Generation Runner: Generates 5 sentences across all 5 model folders.

Runs:
  1. 01_core_ngram_models (Bigram, Trigram, Unigram, Laplace)
  2. 02_keyword_seeded_chatbot (Pride & Prejudice Keyword Bot)
  3. 03_intelligent_austen_chatbot (Victorian Persona Bot)
  4. 04_dailydialog_chatbot (Modern Casual Bot)
  5. 05_smart_hybrid_chatbot (Retrieval-Augmented N-Gram Bot)

Outputs are consolidated to 'deliverables/all_generated_sentences.txt'.
"""

import sys
import subprocess
from pathlib import Path

DELIVERABLES_DIR = Path(__file__).resolve().parent
ROOT_DIR = DELIVERABLES_DIR.parent
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = sys.executable

SCRIPTS = [
    ("01 Core N-Gram Models", DELIVERABLES_DIR / "01_core_ngram_models" / "generate_5_sentences.py"),
    ("02 Keyword-Seeded Chatbot", DELIVERABLES_DIR / "02_keyword_seeded_chatbot" / "generate_5_sentences.py"),
    ("03 Intelligent Austen Chatbot", DELIVERABLES_DIR / "03_intelligent_austen_chatbot" / "generate_5_sentences.py"),
    ("04 DailyDialog Casual Chatbot", DELIVERABLES_DIR / "04_dailydialog_chatbot" / "generate_5_sentences.py"),
    ("05 Smart Hybrid Chatbot", DELIVERABLES_DIR / "05_smart_hybrid_chatbot" / "generate_5_sentences.py"),
]


def main():
    print("\n" + "=" * 80)
    print("  RUNNING MASTER DELIVERABLE GENERATOR ACROSS ALL 5 MODELS")
    print("=" * 80 + "\n")

    all_sections = []

    for name, script_path in SCRIPTS:
        print(f"[*] Executing generator for '{name}'...")
        proc = subprocess.run(
            [str(PYTHON_EXE), str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            print(f"[!] Error in {name}:\n{proc.stderr}")
            continue

        out_txt = script_path.parent / "generated_sentences.txt"
        if out_txt.exists():
            content = out_txt.read_text(encoding="utf-8")
            all_sections.append(content)
            print(f"    [OK] Generated {out_txt.name} ({len(content.splitlines())} lines)")
        else:
            print(f"    [!] Warning: {out_txt} not created")

    master_file = DELIVERABLES_DIR / "all_generated_sentences.txt"
    master_file.write_text("\n\n" + ("#" * 80) + "\n\n".join(all_sections), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"[OK] MASTER EXPORT COMPLETE: {master_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
