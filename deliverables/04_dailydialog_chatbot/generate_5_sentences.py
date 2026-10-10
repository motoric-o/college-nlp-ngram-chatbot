"""Generate 5 conversational sentences for DailyDialogChatbot (Phase 2 Adaptation).

Demonstrates adaptation of the intelligent architecture to modern casual English
trained on 140,000+ utterances from the DailyDialog corpus.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import DailyDialogChatbot


def main():
    print("[*] Initializing DailyDialogChatbot (N=3, Temp=0.8)...")
    bot = DailyDialogChatbot(corpus_name="dailydialog", n=3, temperature=0.8, seed=42)

    prompts = [
        "Hey there! What's up?",
        "I want to order a coffee.",
        "How much does it cost?",               # Coreference: 'it' -> 'coffee'
        "Can you recommend a good movie to watch?",
        "Thanks a lot for your help, see you!",
    ]

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 75)
    log("  DAILYDIALOG CHATBOT: 5 GENERATED MODERN RESPONSES")
    log("  Corpus: DailyDialog (140,368 conversational utterances)")
    log("=" * 75 + "\n")

    for i, prompt in enumerate(prompts, 1):
        reply = bot.respond(prompt)
        log(f"Turn {i}:")
        log(f"  User : {prompt}")
        log(f"  Bot  : {reply}")
        log(f"  [State]: Topic = {bot.last_topic} | Intent = {bot.last_intent}\n")

    out_file = Path(__file__).resolve().parent / "generated_sentences.txt"
    out_file.write_text("\n".join(output_lines), encoding="utf-8")
    log(f"[OK] Output saved to: {out_file}")


if __name__ == "__main__":
    main()
