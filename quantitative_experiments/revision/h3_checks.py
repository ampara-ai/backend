"""Offline controls for the four-condition H3 experiment."""

import json
from pathlib import Path

from quantitative_experiments.revision.h1 import file_hash


def protocol_differences(saved: dict, current: dict) -> list[str]:
    """Keys whose frozen value differs between a saved protocol and the one computed now (empty list = safe to resume).

    The freeze time is ignored. For ``git`` only the commit matters: the output directory itself can make the tree dirty.
    """
    differences = []
    for key in sorted((set(saved) | set(current)) - {"frozen_at_utc"}):
        old, new = saved.get(key), current.get(key)
        if key == "git":
            old, new = (old or {}).get("commit"), (new or {}).get("commit")
        if old != new:
            differences.append(key)
    return differences


def verify_generation(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    for spec in protocol["frozen_inputs"].values():
        if file_hash(spec["path"]) != spec["sha256"]:
            raise ValueError("An H3 input or prompt changed during generation")
    expected_ids = protocol["question_ids"]
    for condition in protocol["conditions"]:
        prov = json.loads((output / "answers" / f"{condition}.provenance.json").read_text())
        if prov["llm"] != protocol["generation_llm"]:
            raise ValueError(f"Generator/prompt/provider settings differ for {condition}")
        if prov["system_prompt_sha256"] != protocol["generation_llm"]["prompts_sha256"]["generation"]:
            raise ValueError(f"Effective generation prompt differs for {condition}")
        if prov["question_ids"] != expected_ids or prov["k"] != 10:
            raise ValueError(f"Question population/context policy differs for {condition}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    verify_generation(parser.parse_args().output)
