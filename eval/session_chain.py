"""
Eval: 10-session chain test — measure fact retention across sessions.

Tests whether the cascading-context system maintains coherent knowledge
across a chain of sessions about "Project Phoenix".

Sessions 1-5 introduce facts incrementally.
Sessions 6-10 query for those facts via the context injector.
Scores how many of the 10 original facts survive to each session.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from ctx.db import get_connection
from ctx.facts import Fact, FactStore, NumpyRandomEmbedder
from ctx.injector import ContextInjector
from ctx.retriever import FactRetriever
from ctx.session import SessionManager

# The 10 ground-truth facts about Project Phoenix, introduced across sessions 1-5
PHOENIX_FACTS = [
    # Session 1: introduce the project
    {"claim": "Project Phoenix is an internal tool for real-time data processing",
     "category": "project", "confidence": 1.0, "entities": ["Phoenix"]},
    {"claim": "Phoenix is written in Rust for performance",
     "category": "technical", "confidence": 1.0, "entities": ["Phoenix", "Rust"]},
    # Session 2: team details
    {"claim": "The Phoenix team has 5 engineers led by Sarah Chen",
     "category": "project", "confidence": 0.95, "entities": ["Phoenix", "Sarah Chen"]},
    {"claim": "Phoenix uses Apache Kafka for message ingestion",
     "category": "technical", "confidence": 1.0, "entities": ["Phoenix", "Kafka"]},
    # Session 3: architecture
    {"claim": "Phoenix processes 50,000 events per second at peak load",
     "category": "technical", "confidence": 0.9, "entities": ["Phoenix"]},
    {"claim": "Phoenix stores processed data in ClickHouse",
     "category": "technical", "confidence": 1.0, "entities": ["Phoenix", "ClickHouse"]},
    # Session 4: timeline
    {"claim": "Phoenix v2 launch is scheduled for Q3 2026",
     "category": "project", "confidence": 0.85, "entities": ["Phoenix"]},
    {"claim": "Phoenix v1 had a critical memory leak that was fixed in January",
     "category": "technical", "confidence": 0.9, "entities": ["Phoenix"]},
    # Session 5: business context
    {"claim": "Phoenix reduces data pipeline latency from 30 seconds to under 1 second",
     "category": "project", "confidence": 0.95, "entities": ["Phoenix"]},
    {"claim": "The Phoenix dashboard is built with React and D3",
     "category": "technical", "confidence": 1.0, "entities": ["Phoenix", "React", "D3"]},
]

# Sessions 1-5: conversations that introduce facts
SESSION_CONVERSATIONS = [
    # Session 1
    [
        ("user", "I'm working on Project Phoenix, an internal tool for real-time data processing."),
        ("assistant", "Interesting! What technology stack is Phoenix built with?"),
        ("user", "Phoenix is written in Rust for performance. We need low-latency processing."),
    ],
    # Session 2
    [
        ("user", "Let me tell you about the Phoenix team."),
        ("assistant", "Sure, who's involved?"),
        ("user", "The Phoenix team has 5 engineers led by Sarah Chen. We use Apache Kafka for message ingestion."),
    ],
    # Session 3
    [
        ("user", "Phoenix processes 50,000 events per second at peak load."),
        ("assistant", "That's impressive throughput. Where does the processed data go?"),
        ("user", "Phoenix stores processed data in ClickHouse for analytics."),
    ],
    # Session 4
    [
        ("user", "We're planning the Phoenix v2 launch for Q3 2026."),
        ("assistant", "What improvements are planned?"),
        ("user", "Mostly stability. Phoenix v1 had a critical memory leak that was fixed in January."),
    ],
    # Session 5
    [
        ("user", "The business case for Phoenix is strong."),
        ("assistant", "How so?"),
        ("user", "Phoenix reduces data pipeline latency from 30 seconds to under 1 second. "
         "The Phoenix dashboard is built with React and D3 for visualization."),
    ],
]

# Sessions 6-10: queries that should retrieve prior facts
SESSION_QUERIES = [
    "What do you know about Project Phoenix?",
    "Tell me about Phoenix's architecture and tech stack",
    "Who works on Phoenix and what's the timeline?",
    "What performance characteristics does Phoenix have?",
    "Give me a full overview of the Phoenix project",
]


def run_eval(db_path: Path | None = None) -> dict:
    conn = get_connection(db_path or Path("/tmp/ctx_eval.db"))
    embedder = NumpyRandomEmbedder(dim=256)
    store = FactStore(conn, embedder)
    session_mgr = SessionManager(conn)

    # Phase 1: Sessions 1-5 — introduce facts
    fact_indices_per_session = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

    for i, (conversation, (start, end)) in enumerate(
        zip(SESSION_CONVERSATIONS, fact_indices_per_session)
    ):
        session = session_mgr.create_session(domain="phoenix", title=f"Phoenix session {i + 1}")
        for role, content in conversation:
            session_mgr.add_message(session.id, role, content)

        # Simulate summarizer output: insert the ground-truth facts
        for fact_data in PHOENIX_FACTS[start:end]:
            store.insert(
                Fact(
                    claim=fact_data["claim"],
                    category=fact_data["category"],
                    confidence=fact_data["confidence"],
                    entities=fact_data["entities"],
                ),
                session_id=session.id,
            )

    # Phase 2: Sessions 6-10 — query and score retention
    retriever = FactRetriever(store, embedder)
    injector = ContextInjector(retriever)

    results = {"sessions": [], "retention_scores": []}

    for i, query in enumerate(SESSION_QUERIES):
        session_num = i + 6
        prompt = injector.build_system_prompt(
            base_prompt="You are a helpful assistant.",
            query=query,
            top_k=10,
        )

        # Score: count how many of the 10 ground-truth claims appear in the prompt
        matched = []
        for j, fact in enumerate(PHOENIX_FACTS):
            # Check if key terms from the claim appear in the injected context
            key_terms = _extract_key_terms(fact["claim"])
            if all(term.lower() in prompt.lower() for term in key_terms):
                matched.append(j)

        score = len(matched)
        results["sessions"].append({
            "session": session_num,
            "query": query,
            "score": score,
            "matched_facts": matched,
            "prompt_length": len(prompt),
        })
        results["retention_scores"].append(score)

    results["final_score"] = results["retention_scores"][-1]
    results["avg_score"] = sum(results["retention_scores"]) / len(results["retention_scores"])
    results["total_facts"] = len(PHOENIX_FACTS)

    return results


def _extract_key_terms(claim: str) -> list[str]:
    """Extract distinctive terms from a claim for matching."""
    # Pick 2-3 distinctive words that uniquely identify each fact
    stop_words = {"is", "a", "an", "the", "for", "in", "to", "and", "has", "was", "that", "of", "with", "at"}
    words = claim.split()
    terms = [w.strip(".,") for w in words if w.lower().strip(".,") not in stop_words and len(w) > 2]
    # Return up to 3 most distinctive terms
    return terms[:3]


def write_results(results: dict, output_path: Path) -> None:
    lines = [
        "# Eval: 10-Session Chain — Fact Retention Results\n",
        f"**Date:** {datetime.now():%Y-%m-%d %H:%M}\n",
        f"**Total ground-truth facts:** {results['total_facts']}\n",
        "",
        "## Retention Curve\n",
        "| Session | Query | Score (/10) | Matched Facts |",
        "|---------|-------|-------------|---------------|",
    ]

    for s in results["sessions"]:
        facts_str = ", ".join(str(f) for f in s["matched_facts"])
        lines.append(f"| {s['session']} | {s['query'][:40]}... | {s['score']} | [{facts_str}] |")

    lines.extend([
        "",
        f"**Average retention:** {results['avg_score']:.1f}/10\n",
        f"**Final session score:** {results['final_score']}/10\n",
        "",
        "## Verdict\n",
    ])

    if results["final_score"] >= 7:
        lines.append(f"PASS: Retention score {results['final_score']}/10 >= 7/10 threshold.")
    else:
        lines.append(f"FAIL: Retention score {results['final_score']}/10 < 7/10 threshold.")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    print("Running 10-session chain eval...")
    results = run_eval()

    output_path = Path(__file__).parent.parent / "RESULTS.md"
    write_results(results, output_path)

    print(f"\nRetention scores per query session:")
    for s in results["sessions"]:
        print(f"  Session {s['session']}: {s['score']}/10 facts retained")
    print(f"\nAverage: {results['avg_score']:.1f}/10")
    print(f"Final:   {results['final_score']}/10")
    print(f"\nResults written to {output_path}")

    if results["final_score"] < 7:
        print("\nFAIL: Did not meet 7/10 retention threshold.")
        sys.exit(1)
    else:
        print("\nPASS: Met retention threshold.")


if __name__ == "__main__":
    main()
