"""Generate 5 conversational sentences for KeywordSeededChatbot (Phase 1).

Demonstrates topic-anchored sampling from Jane Austen's 'Pride and Prejudice'.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import KeywordSeededChatbot


def main():
    print("[*] Initializing KeywordSeededChatbot (N=3, Temp=0.8)...")
    bot = KeywordSeededChatbot(corpus_name="pride", n=3, temperature=0.8, seed=42)

    prompts = [
        "What do you think of Mr. Darcy?",
        "Tell me about marriage in Hertfordshire.",
        "Have you seen Miss Elizabeth Bennet?",
        "Where is Mr. Bingley staying?",
        "What are your thoughts on Pemberley?",
    ]

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 75)
    log("  KEYWORD-SEEDED CHATBOT: 5 GENERATED CONVERSATIONAL RESPONSES")
    log("  Model: Trigram (N=3) | Sampling Temperature: 0.80")
    log("  Corpus: Jane Austen's 'Pride and Prejudice'")
    log("=" * 75 + "\n")

    for i, prompt in enumerate(prompts, 1):
        reply = bot.respond(prompt)
        log(f"Turn {i}:")
        log(f"  User : {prompt}")
        log(f"  Bot  : {reply}\n")

    out_file = Path(__file__).resolve().parent / "generated_sentences.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"[OK] Output saved to: {out_file}")


if __name__ == "__main__":
    main()
