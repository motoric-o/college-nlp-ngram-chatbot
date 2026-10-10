"""Generate 5 conversational sentences for SmartChatbot (Phase 3).

Demonstrates Hybrid Retrieval-Augmented N-Gram Generation:
  1. Dialogue turn-pair inverted index (76,052 pairs)
  2. N-Gram generative completion from dialogue openers
  3. Multi-criteria re-ranking with topic overlap and fluency
  4. Coreference memory across multi-turn inquiries
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import SmartChatbot


def main():
    print("[*] Initializing SmartChatbot (Phase 3 Hybrid Architecture)...")
    bot = SmartChatbot(corpus_name="dailydialog", n=3, temperature=0.7, seed=42)

    prompts = [
        "Hello! How are you doing today?",
        "I want to buy a computer.",            # Addresses Screenshot Turn 1
        "How much does it cost?",               # Addresses Screenshot Turn 2 ('it' -> computer)
        "Can you recommend a good laptop?",
        "Thank you so much, that was very helpful!",
    ]

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 75)
    log("  SMART HYBRID CHATBOT: 5 GENERATED RESPONSES")
    log("  Architecture: Retrieval-Augmented N-Gram Fusion (DailyDialog)")
    log("=" * 75 + "\n")

    for i, prompt in enumerate(prompts, 1):
        reply = bot.respond(prompt)
        log(f"Turn {i}:")
        log(f"  User  : {prompt}")
        log(f"  Smart : {reply}")
        log(f"  [State]: Topic = {bot.last_topic} | Intent = {bot.last_intent}\n")

    out_file = Path(__file__).resolve().parent / "generated_sentences.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"[OK] Output saved to: {out_file}")


if __name__ == "__main__":
    main()
