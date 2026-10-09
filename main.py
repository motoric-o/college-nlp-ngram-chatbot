"""Interactive N-Gram Chatbot Application.

An interactive terminal interface to converse with statistical N-gram language models
trained on Jane Austen's 'Pride and Prejudice' (or any configured corpus).

Usage:
    python main.py
    python main.py --quick
    python main.py --temp 0.5 --model 3
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path so 'src' is always importable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import CHATBOT_REGISTRY, KeywordSeededChatbot


def print_banner(bot):
    model_name = "Trigram (N=3)" if bot.n == 3 else "Bigram (N=2)"
    bot_class_name = bot.__class__.__name__
    print("\n" + "=" * 70)
    print("  N-GRAM CHATBOT: JANE AUSTEN (Pride & Prejudice)")
    print(f"  Bot: {bot_class_name} | Model: {model_name} | Temp: {bot.temperature:.2f}")
    print("=" * 70)
    print("Type your message and press Enter.")
    print("Special commands:")
    print("  /temp <float>   Set sampling temperature (e.g., /temp 0.5 or /temp 1.2)")
    print("  /model <2|3>    Switch between Bigram (2) and Trigram (3)")
    print("  /bot <name>     Switch chatbot architecture (available: keyword, intelligent)")
    print("  /history        Show conversational memory (intelligent bot)")
    print("  /info           Show current model configuration")
    print("  /help           Show available commands")
    print("  exit / quit     Exit the conversation")
    print("=" * 70 + "\n")


def print_help():
    print("\nAvailable in-chat commands:")
    print("  /temp <val>   Set sampling temperature (> 0.0). Lower = safer, Higher = creative.")
    print("  /model <2|3>  Switch between 2 (Bigram) and 3 (Trigram).")
    print("  /bot <name>   Switch chatbot class (available: 'keyword', 'intelligent').")
    print("  /history      Show remembered entity, last intent and recent turns.")
    print("  /info         Show current model configuration and corpus vocabulary.")
    print("  /help         Show this help message.")
    print("  exit / quit   Leave the conversation.\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive N-Gram Chatbot")
    parser.add_argument(
        "--corpus",
        default="pride",
        choices=["pride", "pride_and_prejudice", "brown", "reuters"],
        help="Corpus to train on (default: 'pride')",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=3,
        choices=[2, 3],
        help="N-gram order (2 for bigram, 3 for trigram; default: 3)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--bot",
        default="intelligent",
        choices=list(CHATBOT_REGISTRY.keys()),
        help="Chatbot architecture to use (default: 'intelligent')",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Start quickly with sample sentences",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print(f"[*] Initializing {args.bot.capitalize()} Chatbot on corpus '{args.corpus}'...")
    bot_class = CHATBOT_REGISTRY[args.bot]
    bot = bot_class(
        corpus_name=args.corpus,
        n=args.model,
        temperature=args.temp,
        quick=args.quick,
        seed=args.seed,
    )

    print_banner(bot)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAusten-Bot: I take my leave of you. Farewell!\n")
            break

        if not user_input:
            continue

        lower_input = user_input.lower()
        if lower_input in ("exit", "quit", "q", ":q"):
            print("\nAusten-Bot: Farewell! It was an honour conversing with you.\n")
            break

        # Command handling
        if lower_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print_help()
                continue

            elif cmd == "/temp":
                if len(parts) < 2:
                    print(f"Current temperature: {bot.temperature:.2f}. Usage: /temp <float>")
                else:
                    try:
                        val = float(parts[1])
                        bot.set_temperature(val)
                        print(f"Sampling temperature updated to {bot.temperature:.2f}")
                    except ValueError:
                        print("Invalid temperature value. Please provide a positive number.")
                continue

            elif cmd == "/model":
                if len(parts) < 2:
                    print(f"Current model order: N={bot.n}. Usage: /model 2 or /model 3")
                else:
                    try:
                        val = int(parts[1])
                        bot.set_model_order(val)
                        name = "Trigram (N=3)" if bot.n == 3 else "Bigram (N=2)"
                        print(f"Model updated to {name}")
                    except ValueError as e:
                        print(f"Error: {e}")
                continue

            elif cmd == "/bot":
                if len(parts) < 2:
                    print(f"Current bot: {bot.__class__.__name__}. Available: {list(CHATBOT_REGISTRY.keys())}")
                else:
                    target_bot = parts[1].lower()
                    if target_bot in CHATBOT_REGISTRY:
                        bot_class = CHATBOT_REGISTRY[target_bot]
                        bot = bot_class(
                            corpus_name=args.corpus,
                            n=bot.n,
                            temperature=bot.temperature,
                            quick=args.quick,
                            seed=args.seed,
                        )
                        print(f"Switched chatbot to {bot.__class__.__name__}")
                    else:
                        print(f"Unknown bot '{target_bot}'. Available: {list(CHATBOT_REGISTRY.keys())}")
                continue

            elif cmd == "/info":
                model_name = "Trigram (N=3)" if bot.n == 3 else "Bigram (N=2)"
                print(f"\nConfiguration:")
                print(f"  Chatbot class : {bot.__class__.__name__}")
                print(f"  Corpus        : {bot.corpus_name}")
                print(f"  N-gram model  : {model_name}")
                print(f"  Temperature   : {bot.temperature:.2f}")
                print(f"  Vocabulary    : {len(bot.vocabulary):,} words\n")
                continue

            elif cmd == "/history":
                if not hasattr(bot, "history"):
                    print("This chatbot has no conversational memory.")
                else:
                    print(f"\nRemembered entity: {getattr(bot, 'last_entity', None)}")
                    print(f"Last intent      : {getattr(bot, 'last_intent', None)}")
                    for i, (u, r) in enumerate(bot.history[-5:], 1):
                        print(f"  [{i}] You: {u}\n      Bot: {r}")
                    print()
                continue

            else:
                print(f"Unknown command '{cmd}'. Type /help for available commands.")
                continue

        # Generate chatbot reply
        reply = bot.respond(user_input)
        print(f"Austen-Bot: {reply}\n")


if __name__ == "__main__":
    main()
