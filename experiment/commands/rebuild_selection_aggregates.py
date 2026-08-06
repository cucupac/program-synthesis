"""Rebuild ignored formal aggregates from their tracked cell files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiment import budget_intervention
from experiment import capacity_curve as capacity
from experiment.commands.run_k_sweep_experiment import write_json_atomic


ROOT = Path(__file__).resolve().parents[2]
BUDGET_AGGREGATE_SHA256 = (
    "acabebabac6266745405406ada9cb832da7e0e110db7ba84adddfc65c55903e8"
)
AGGREGATES = (
    (
        Path("experiment/data/selection/capacity_curve"),
        "full_selection_experiment_capacity_curve.json",
        capacity.EXPERIMENT_NAME,
        budget_intervention.FORMAL_SOURCE_SHA256,
    ),
    (
        Path("experiment/data/selection/budget_intervention"),
        "full_selection_experiment_budget_intervention.json",
        budget_intervention.EXPERIMENT_NAME,
        BUDGET_AGGREGATE_SHA256,
    ),
)


def build_payload(directory: Path, experiment_name: str) -> dict:
    registration = json.loads(
        (directory / "registration.json").read_text(encoding="utf-8")
    )
    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (directory / "cells").glob("*.json")
    ]
    cells.sort(
        key=lambda cell: (
            cell["seed"],
            capacity.CONDITIONS.index(cell["condition"]),
        )
    )
    return {
        "experiment_name": experiment_name,
        "smoke": False,
        "registration": registration,
        "cells": cells,
    }


def payload_sha256(payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def rebuild(
    directory: Path, output_name: str, experiment_name: str, expected: str
) -> None:
    payload = build_payload(directory, experiment_name)
    actual = payload_sha256(payload)
    if actual != expected:
        raise RuntimeError(
            f"refusing to rebuild {output_name}: expected {expected}, got {actual}"
        )
    output = directory / output_name
    write_json_atomic(output, payload)
    print(f"rebuilt {output.relative_to(ROOT)} sha256={actual}")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    for relative_directory, output_name, experiment_name, expected in AGGREGATES:
        rebuild(ROOT / relative_directory, output_name, experiment_name, expected)


if __name__ == "__main__":
    main()
