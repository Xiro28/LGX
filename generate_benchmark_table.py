#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FACT_RE = re.compile(r"[^.]+\.")


@dataclass(frozen=True)
class RowSpec:
    pipeline: str
    guard_enforcement: str
    filename: str


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    dataset_path: Path
    filename_prefix: str
    ignored_predicates: frozenset[str]


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        label="G",
        dataset_path=Path("datasets/dataset_graph.json"),
        filename_prefix="graph",
        ignored_predicates=frozenset(),
    ),
    BenchmarkSpec(
        label="LNRS",
        dataset_path=Path("datasets/dataset_final_lnrs.json"),
        filename_prefix="lnrs",
        ignored_predicates=frozenset({"problem_name"}),
    ),
)


MODELS: tuple[tuple[str, str], ...] = (
    ("llama3.1_8b", "Small (8B)"),
    ("llama3.1_70b", "Medium (70B)"),
    ("gpt-oss_120b", "Large (120B)"),
)


ROWS: tuple[RowSpec, ...] = (
    RowSpec("LLMASP", "None", "{prefix}_llmasp_base.json"),
    RowSpec("LLMASP", "A posteriori", "{prefix}_llmasp_g.json"),
    RowSpec("LGX", "By design", "{prefix}_lgx_base_cache_adv.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize experiment JSON files into a terminal table."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiment_results"),
        help="Directory that contains per-model benchmark JSON files.",
    )
    parser.add_argument(
        "--format",
        choices=("ascii", "markdown"),
        default="ascii",
        help="Output format.",
    )
    return parser.parse_args()


def normalize_fact(raw_fact: str) -> str:
    return "".join(raw_fact.split())


def parse_facts(text: str) -> set[str]:
    if not text or text == "TIMEOUT":
        return set()

    flattened = text.replace("\n", " ")
    return {normalize_fact(match.group(0)) for match in FACT_RE.finditer(flattened)}


def predicate_name(fact: str) -> str:
    return fact.split("(", 1)[0]


def filter_facts(facts: set[str], ignored_predicates: frozenset[str]) -> set[str]:
    if not ignored_predicates:
        return facts
    return {fact for fact in facts if predicate_name(fact) not in ignored_predicates}


def instance_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0

    true_positives = len(predicted & gold)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted)
    recall = true_positives / len(gold)
    return 2 * precision * recall / (precision + recall)


def load_grouped_dataset(path: Path) -> dict[str, list[dict[str, str]]]:
    dataset = json.loads(path.read_text())
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in dataset:
        grouped[item["problem_name"]].append(item)
    return grouped


def total_llm_calls(payload: dict) -> int | None:
    total = 0
    found = False
    for section_name, section in payload.items():
        if section_name == "dataset":
            continue
        if isinstance(section, dict) and "llm_calls" in section:
            total += section["llm_calls"]
            found = True
    return total if found else None


def count_nonempty_predictions(payload: dict) -> int:
    total = 0
    for section_name, section in payload.items():
        if section_name == "dataset":
            continue
        if not isinstance(section, dict):
            continue
        for item in section.get("results", []):
            if item and item != "TIMEOUT":
                total += 1
    return total


def evaluate_result(
    grouped_dataset: dict[str, list[dict[str, str]]],
    payload: dict,
    ignored_predicates: frozenset[str],
) -> tuple[float, float, int, list[str]]:
    scores: list[float] = []
    perfect_matches = 0
    warnings: list[str] = []

    for section_name, section in payload.items():
        if section_name == "dataset":
            continue

        if section_name not in grouped_dataset:
            warnings.append(f"Unexpected section '{section_name}' in results.")
            continue

        gold_items = grouped_dataset[section_name]
        predicted_items = section.get("results", [])

        if len(predicted_items) != len(gold_items):
            warnings.append(
                f"Section '{section_name}' has {len(predicted_items)} predictions but {len(gold_items)} gold items."
            )

        for predicted_text, gold_item in zip(predicted_items, gold_items, strict=False):
            predicted = filter_facts(parse_facts(predicted_text), ignored_predicates)
            gold = filter_facts(parse_facts(gold_item["output"]), ignored_predicates)
            scores.append(instance_f1(predicted, gold))
            if predicted == gold:
                perfect_matches += 1

    count = len(scores)
    mean_f1 = sum(scores) / count if count else 0.0
    perfect_rate = perfect_matches / count if count else 0.0
    return mean_f1, perfect_rate, count, warnings


def format_score(value: float) -> str:
    return f"{value:.3f}"


def format_calls(value: int | None) -> str:
    return str(value) if value is not None else "n/a"


def render_ascii(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def render_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    parts = [render_row(headers), separator]
    parts.extend(render_row(row) for row in rows)
    return "\n".join(parts)


def render_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    headers = [
        "Bench.",
        "O",
        "Pipeline",
        "O Calls",
        "Guard Enforcement",
        "F1-Score",
        "Perfect Rate",
    ]

    rows: list[list[str]] = []
    notices: list[str] = []

    for benchmark in BENCHMARKS:
        grouped_dataset = load_grouped_dataset(benchmark.dataset_path)
        benchmark_label = benchmark.label

        for model_dir, model_label in MODELS:
            size_label = model_label

            for row_spec in ROWS:
                result_path = (
                    args.results_dir
                    / model_dir
                    / row_spec.filename.format(prefix=benchmark.filename_prefix)
                )

                if not result_path.exists():
                    notices.append(f"Missing result file: {result_path}")
                    rows.append(
                        [
                            benchmark_label,
                            size_label,
                            row_spec.pipeline,
                            "missing",
                            row_spec.guard_enforcement,
                            "missing",
                            "missing",
                        ]
                    )
                else:
                    payload = json.loads(result_path.read_text())
                    mean_f1, perfect_rate, _, warnings = evaluate_result(
                        grouped_dataset,
                        payload,
                        benchmark.ignored_predicates,
                    )
                    llm_calls = total_llm_calls(payload)
                    nonempty_predictions = count_nonempty_predictions(payload)
                    if llm_calls is None:
                        notices.append(
                            f"{result_path} does not contain llm_calls; the table shows 'n/a' for O Calls."
                        )
                    if nonempty_predictions == 0:
                        notices.append(
                            f"{result_path} contains no non-empty predictions; F1 and perfect rate will stay at 0.000."
                        )
                    if warnings:
                        notices.extend(f"{result_path}: {warning}" for warning in warnings)

                    rows.append(
                        [
                            benchmark_label,
                            size_label,
                            row_spec.pipeline,
                            format_calls(llm_calls),
                            row_spec.guard_enforcement,
                            format_score(mean_f1),
                            format_score(perfect_rate),
                        ]
                    )

                benchmark_label = ""
                size_label = ""

    if args.format == "ascii":
        print(render_ascii(headers, rows))
    else:
        print(render_markdown(headers, rows))

    if notices:
        print("\nNotes:")
        seen = set()
        for notice in notices:
            if notice in seen:
                continue
            seen.add(notice)
            print(f"- {notice}")


if __name__ == "__main__":
    main()
