#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "campus_qa.json"
MIN_RECORDS = 200


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("campus_qa.json must be a JSON list")

    errors: list[str] = []
    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            errors.append(f"row {index}: item is not an object")
            continue

        for field in ["id", "category", "instruction", "output"]:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"row {index}: missing or empty {field}")

        record_id = item.get("id")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(f"row {index}: duplicate id {record_id}")
            seen_ids.add(record_id)

        input_value = item.get("input", "")
        if not isinstance(input_value, str):
            errors.append(f"row {index}: input must be a string")

        category = item.get("category")
        if isinstance(category, str) and category.strip():
            categories[category] += 1

    if len(data) < MIN_RECORDS:
        errors.append(f"dataset has {len(data)} records, expected at least {MIN_RECORDS}")

    print(f"dataset: {DATA_PATH}")
    print(f"records: {len(data)}")
    print(f"categories: {len(categories)}")
    for category, count in sorted(categories.items()):
        print(f"- {category}: {count}")

    if errors:
        print("validation: FAILED")
        for error in errors[:50]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("validation: PASSED")


if __name__ == "__main__":
    main()
