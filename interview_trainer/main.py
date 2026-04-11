"""
main.py
-------
Entry point for the AI Interview Trainer RL System.

Usage
-----
  # Interactive mode (prompts for question & topic)
  python main.py

  # Run a preset demo question
  python main.py --demo

  # Custom question
  python main.py --question "Explain object-oriented programming" --topic "python" --attempts 4

Environment
-----------
  Set your Gemini API key before running:
    export GEMINI_API_KEY="your-key-here"    # Linux / macOS
    set GEMINI_API_KEY=your-key-here         # Windows CMD
"""

import argparse
import json
import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rl_environment import InterviewEnv, EpisodeResult

# ── Topic options ─────────────────────────────────────────────────────────────

TOPICS = [
    "python", "machine learning", "data structures",
    "system design", "behavioral", "general"
]

DEMO_QUESTIONS = [
    {"question": "Explain how a hash table works under the hood, including collision resolution.", "topic": "data structures"},
    {"question": "Describe the difference between processes and threads.", "topic": "system design"},
    {"question": "What is the difference between a list and a tuple in Python?", "topic": "python"}
]


# ── Pretty printers ───────────────────────────────────────────────────────────

def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║        🎓  AI Interview Trainer  —  RL System        ║
║    Hybrid NLP + LLM evaluation with reward loop      ║
╚══════════════════════════════════════════════════════╝
""")


def print_step_detail(step, max_reward: float):
    print(f"\n  ┌─ Attempt {step.attempt} Detail {'─'*35}")
    print(f"  │  Score   : {step.llm_score}/10")

    import textwrap
    print(f"  │\n  │  Generated Answer:")
    answer_lines = step.answer.splitlines()
    for line in answer_lines:
        for wrapped_line in textwrap.wrap(line, width=60):
            print(f"  │    {wrapped_line}")

    if step.llm_feedback:
        print(f"  │\n  │  Evaluator Feedback:")
        feedback_lines = step.llm_feedback.splitlines()
        for f_line in feedback_lines:
            for wrapped_f in textwrap.wrap(f_line, width=60):
                print(f"  │    {wrapped_f}")

    print(f"  └{'─'*47}")


    # JSON results functionality removed


# ── Interactive prompts ───────────────────────────────────────────────────────

def prompt_question() -> tuple[str, str]:
    print("  Available topics:")
    for i, t in enumerate(TOPICS, 1):
        print(f"    {i}. {t}")
    topic_input = input("\n  Enter topic number or name [default: general]: ").strip()
    if topic_input.isdigit():
        idx   = int(topic_input) - 1
        topic = TOPICS[idx] if 0 <= idx < len(TOPICS) else "general"
    else:
        topic = topic_input.lower() if topic_input else "general"

    question = input("\n  Enter your interview question:\n  > ").strip()
    if not question:
        question = "Tell me about yourself and your technical background."
    return question, topic


def prompt_attempts() -> int:
    raw = input("\n  Number of attempts (1–5) [default: 3]: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= 5:
        return int(raw)
    return 3


# ── Main ──────────────────────────────────────────────────────────────────────

def run(question: str, topic: str, n_attempts: int, save: bool = True):
    env = InterviewEnv(
        question   = question,
        topic      = topic,
        n_attempts = n_attempts,
        verbose    = True,
    )
    episode = env.run_episode()

    # Detailed step breakdown
    print("\n\n📋  DETAILED BREAKDOWN")
    for step in episode.steps:
        print_step_detail(step, env.reward_engine.max_reward)

    print("\n\n📈  SCORE PROGRESSION")
    for step in episode.steps:
        arrow = "" if step.attempt == 1 else (
            " ▲" if step.delta > 0 else (" ▼" if step.delta < 0 else " ●")
        )
        print(f"  Attempt {step.attempt}: {step.llm_score}/10{arrow}")

    # JSON save removed

    return episode


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="AI Interview Trainer RL System")
    parser.add_argument("--demo",     action="store_true",
                        help="Run a preset demo question")
    parser.add_argument("--question", type=str, default="",
                        help="Interview question to answer")
    parser.add_argument("--topic",    type=str, default="general",
                        help=f"Topic domain: {', '.join(TOPICS)}")
    parser.add_argument("--attempts", type=int, default=3,
                        help="Number of RL iterations (1-5)")
    parser.add_argument("--no-save",  action="store_true",
                        help="Don't save results.json")
    args = parser.parse_args()

    # ── API key notice ────────────────────────────────────────────────────────
    # Key is already integrated into the environment.
    pass

    # ── Mode selection ────────────────────────────────────────────────────────
    if args.demo:
        import random
        choice = random.choice(DEMO_QUESTIONS)
        question, topic = choice["question"], choice["topic"]
        n_attempts = 3
        print(f"  🎲 Demo question: {question}")
    elif args.question:
        question   = args.question
        topic      = args.topic
        n_attempts = max(1, min(5, args.attempts))
    else:
        # Interactive
        question, topic = prompt_question()
        n_attempts = prompt_attempts()

    run(question, topic, n_attempts, save=not args.no_save)


if __name__ == "__main__":
    main()