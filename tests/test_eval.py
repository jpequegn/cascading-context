from pathlib import Path

from eval.session_chain import run_eval


def test_session_chain_retention(tmp_path: Path):
    results = run_eval(db_path=tmp_path / "eval.db")

    assert results["total_facts"] == 10
    assert len(results["retention_scores"]) == 5

    # Acceptance criteria: final score >= 7/10
    assert results["final_score"] >= 7, (
        f"Retention score {results['final_score']}/10 below 7/10 threshold"
    )

    # Each session should retrieve at least some facts
    for s in results["sessions"]:
        assert s["score"] > 0
