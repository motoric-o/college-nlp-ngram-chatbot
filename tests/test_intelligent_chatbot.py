"""Verification script for IntelligentChatbot (Phase 2).

Usage:
    python test_intelligent_chatbot.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chatbot import CHATBOT_REGISTRY, IntelligentChatbot


def main():
    bot = IntelligentChatbot(corpus_name="pride", n=3, temperature=0.8, seed=7)

    # 1. Intent rules
    assert bot.classify_intent("Hello there")[0] == "GREETING"
    assert bot.classify_intent("Who are you?")[0] == "IDENTITY"
    assert bot.classify_intent("Goodbye!")[0] == "FAREWELL"
    intent, topic = bot.classify_intent("What do you think of Mr. Darcy?")
    assert intent == "OPINION" and "Darcy" in topic
    assert bot.respond("Hello!") in bot.INTENT_RESPONSES["GREETING"]
    print("[PASS] Intent classification")

    # 2. Memory + coreference
    bot.respond("Tell me about Mr. Darcy")
    assert bot.last_entity == "darcy", bot.last_entity
    assert bot.resolve_coreference(["is", "he", "proud"]) == ["is", "darcy", "proud"]
    bot.respond("Is he proud?")
    assert bot.last_entity == "darcy"
    print("[PASS] Conversational memory & coreference")

    # 3. Re-ranking penalises dangling / short candidates
    good = "darcy was a proud man".split()
    bad = "darcy walked to the".split()
    assert bot.score_candidate(good, {"darcy"}, True) > bot.score_candidate(bad, {"darcy"}, False)
    print("[PASS] Candidate re-ranking")

    # 4. Reflection
    assert bot.reflect("I love my sister") == "you love your sister"
    print("[PASS] ELIZA reflection")

    # 5. Registry
    assert set(CHATBOT_REGISTRY) >= {"keyword", "intelligent"}
    print("[PASS] Registry")

    print("\nSample conversation:")
    for q in ["Good morning!", "What do you think of marriage?", "Tell me about Elizabeth",
              "Does she like dancing?", "I feel rather tired today", "What is a computer?",
              "Who are you?", "Farewell"]:
        print(f"You: {q}\nBot: {bot.respond(q)}\n")


if __name__ == "__main__":
    main()
