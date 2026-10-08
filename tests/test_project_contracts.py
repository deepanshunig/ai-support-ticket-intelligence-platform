"""Keep authored evaluation and explicit routing configuration internally consistent."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_authored_challenges_preserve_partition_and_multi_intent_contracts():
    mapping = json.loads((ROOT / "configs/team_mapping.json").read_text(encoding="utf-8"))
    intent_to_team = mapping["intent_to_team"]
    rows = [
        json.loads(line)
        for line in (ROOT / "data/challenge_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({" ".join(row["text"].lower().split()) for row in rows}) == len(rows)
    groups = {
        "clearly_unrelated",
        "nearby_unsupported_banking",
        "ambiguous_supported",
        "multi_intent",
    }
    counts = Counter((row["group"], row["partition"]) for row in rows)
    assert counts == Counter(
        {
            (group, partition): count
            for group in groups
            for partition, count in [("policy_development", 12), ("heldout", 13)]
        }
    )
    for row in rows:
        assert row["expected_action"] == "needs_review" and row["rationale"].strip()
        if row["group"] == "multi_intent":
            intents = row["expected_intents"]
            assert len(intents) == 2 and len(set(intents)) == 2
            assert len({intent_to_team[intent] for intent in intents}) == 2


def test_demo_mapping_has_full_and_explicit_declared_team_coverage():
    mapping = json.loads((ROOT / "configs/team_mapping.json").read_text(encoding="utf-8"))
    assert len(mapping["intent_to_team"]) == 77
    declared = {team["id"] for team in mapping["teams"]}
    assert len(declared) == 5
    assert set(mapping["intent_to_team"].values()) == declared
