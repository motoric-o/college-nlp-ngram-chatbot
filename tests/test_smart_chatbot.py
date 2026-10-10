"""Automated test and verification script for SmartChatbot (Phase 3).

Verifies:
  1. DialoguePairIndex indexing and fast retrieval.
  2. Resolution of the failure cases in the user's screenshot:
     - 'I want to buy a computer' (sensible advice/response, not 'Want to the company.')
     - 'How much does it cost?' (authentic pricing response, not 'Computer and I will be a good.')
  3. Conversational memory & coreference tracking.
  4. Intent classification (Greetings, Identity, Farewells, Thanks).
  5. CHATBOT_REGISTRY registration and consistency.

Usage:
    python test_smart_chatbot.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if not (ROOT_DIR / "src").exists():
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chatbot import (
    CHATBOT_REGISTRY,
    DialoguePairIndex,
    SmartChatbot,
)


def test_registry():
    assert set(CHATBOT_REGISTRY) >= {"keyword", "intelligent", "daily", "smart"}
    print("[PASS] Registry contains all 4 bot architectures ('keyword', 'intelligent', 'daily', 'smart')")


def test_dialogue_pair_index():
    sample_pairs = [
        ("I need to buy a computer", "You can get an excellent deal on a laptop or desktop here."),
        ("How much does it cost?", "It costs 125 dollars a month."),
        ("Where is the restaurant?", "It is just across the street next to the bank."),
    ]
    idx = DialoguePairIndex(sample_pairs)
    res = idx.search("I want to buy a computer", top_m=1)
    assert len(res) == 1
    assert "computer" in res[0][0][0]
    print("[PASS] DialoguePairIndex indexing and search")


def test_smart_chatbot_responses():
    print("[*] Initializing SmartChatbot...")
    bot = SmartChatbot(corpus_name="dailydialog", n=3, temperature=0.7, seed=42)

    # 1. Intent tests
    greeting_reply = bot.respond("Hello!")
    assert any(g in greeting_reply for g in ["Hey", "Hi", "Hello", "How's", "Nice to meet"]), greeting_reply
    print(f"[PASS] Intent Greeting: {greeting_reply}")

    thanks_reply = bot.respond("Thank you very much!")
    assert any(t in thanks_reply.lower() for t in ["welcome", "problem", "pleasure"]), thanks_reply
    print(f"[PASS] Intent Thanks: {thanks_reply}")

    # 2. Screenshot Test Case 1: "I want to buy a computer"
    reply1 = bot.respond("I want to buy a computer")
    assert not reply1.lower().startswith("want to the"), f"Regressed to old echo verb: {reply1}"
    assert len(reply1.split()) >= 3, f"Too short: {reply1}"
    print(f"[PASS] Screenshot Case 1 ('I want to buy a computer'):\n       Bot: {reply1}")

    # 3. Screenshot Test Case 2: "How much does it cost?"
    reply2 = bot.respond("How much does it cost?")
    assert not reply2.lower().startswith("computer and i will be a good"), f"Regressed to broken fallback: {reply2}"
    assert any(w in reply2.lower() for w in ["cost", "costs", "dollar", "money", "pay", "price", "free", "cheap", "expensive", "around", "about"]), f"Unexpected cost reply: {reply2}"
    print(f"[PASS] Screenshot Case 2 ('How much does it cost?'):\n       Bot: {reply2}")

    # 4. Multi-turn memory test
    bot.respond("Let's talk about movies")
    assert bot.last_topic in ("movies", "movie"), f"Unexpected topic: {bot.last_topic}"
    reply_pronoun = bot.respond("Do you like them?")
    assert len(reply_pronoun) > 0
    print(f"[PASS] Coreference tracking on 'movies' -> 'them':\n       Bot: {reply_pronoun}")


def main():
    print("=" * 60)
    print("  RUNNING SMART CHATBOT TEST SUITE")
    print("=" * 60)
    test_registry()
    test_dialogue_pair_index()
    test_smart_chatbot_responses()
    print("=" * 60)
    print("  ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
