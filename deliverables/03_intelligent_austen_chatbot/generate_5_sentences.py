"""Generate 5 conversational sentences for IntelligentChatbot (Phase 2).

Demonstrates intent rules, character coreference memory ('he' -> Darcy),
and Best-of-K re-ranking with Victorian persona framing.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import IntelligentChatbot


def main():
    print("[*] Initializing IntelligentChatbot (N=3, Temp=0.8, Best-of-5)...")
    bot = IntelligentChatbot(corpus_name="pride", n=3, temperature=0.8, seed=42)

    prompts = [
        "Good day! How do you do?",
        "What do you think of Mr. Darcy?",
        "Is he proud?",                         # Coreference: 'he' -> 'darcy'
        "I feel rather fatigued by this journey.", # ELIZA pronoun reflection
        "Farewell, I must take my leave.",
    ]

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 75)
    log("  INTELLIGENT AUSTEN CHATBOT: 5 GENERATED RESPONSES")
    log("  Features: Intent Classification, Coreference Memory, Best-of-5 Scoring")
    log("  Corpus: Jane Austen's 'Pride and Prejudice'")
    log("=" * 75 + "\n")

    for i, prompt in enumerate(prompts, 1):
        reply = bot.respond(prompt)
        log(f"Turn {i}:")
        log(f"  User   : {prompt}")
        log(f"  Austen : {reply}")
        log(f"  [State]: Last Entity = {bot.last_entity} | Last Intent = {bot.last_intent}\n")

    out_file = Path(__file__).resolve().parent / "generated_sentences.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"[OK] Output saved to: {out_file}")


if __name__ == "__main__":
    main()
