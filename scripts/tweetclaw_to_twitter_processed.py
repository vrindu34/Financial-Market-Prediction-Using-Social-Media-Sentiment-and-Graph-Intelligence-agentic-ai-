#!/usr/bin/env python3
"""Convert reviewed TweetClaw exports into the app's Twitter input CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TEXT_KEYS = ("text", "tweet", "content", "body", "full_text")
STOCK_KEYS = ("stock", "symbol", "ticker")
TIME_KEYS = ("created_at", "createdAt", "timestamp", "time", "date")
AUTHOR_KEYS = ("author", "username", "screen_name", "user")
URL_KEYS = ("url", "permalink", "link")
ID_KEYS = ("id", "tweet_id", "tweetId")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw JSON, JSONL, or CSV exports to twitter_processed.csv.",
    )
    parser.add_argument("input", type=Path, help="TweetClaw export path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("twitter_processed.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--stock",
        default="",
        help="Fallback stock ticker when the export has no stock, symbol, or ticker field",
    )
    return parser.parse_args()


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def unwrap_record(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    for key in ("tweet", "post", "item", "record"):
        nested = value.get(key)
        if isinstance(nested, dict):
            merged = dict(nested)
            for parent_key in ("query", "stock", "symbol", "ticker"):
                if parent_key in value and parent_key not in merged:
                    merged[parent_key] = value[parent_key]
            return merged
    return value


def records_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [record for item in value if (record := unwrap_record(item))]
    if isinstance(value, dict):
        for key in ("tweets", "posts", "results", "items", "data", "records"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [record for item in nested if (record := unwrap_record(item))]
        record = unwrap_record(value)
        return [record] if record else []
    return []


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        return records_from_json(json.loads(text))
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSONL on line {line_number}: {error}") from error
            record = unwrap_record(value)
            if record:
                records.append(record)
        return records


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_json_or_jsonl(path)


def normalize_records(records: list[dict[str, Any]], fallback_stock: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        text = first_value(record, TEXT_KEYS)
        if not text:
            continue
        row = {
            "stock": first_value(record, STOCK_KEYS) or fallback_stock,
            "text": text,
            "created_at": first_value(record, TIME_KEYS),
            "author": first_value(record, AUTHOR_KEYS),
            "url": first_value(record, URL_KEYS),
            "source_id": first_value(record, ID_KEYS),
            "source": "tweetclaw",
        }
        dedupe_key = row["source_id"] or row["url"] or f"{row['stock']}:{row['text']}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(row)
    return rows


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["stock", "text", "created_at", "author", "url", "source_id", "source"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    rows = normalize_records(records, args.stock.strip().upper())
    if not rows:
        raise SystemExit("No tweet text found in input export.")
    write_output(args.output, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
