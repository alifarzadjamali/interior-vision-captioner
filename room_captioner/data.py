"""Download a small, public interior design dataset."""

from __future__ import annotations

import json
from pathlib import Path

DATASET_ID = "ellljoy/interior-design"
DATASET_SPLIT = "train"


def download_dataset(
    output_dir: str | Path = "data",
    limit: int = 12,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    """Save dataset images locally and write their reference captions as JSONL."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    from datasets import load_dataset

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    reference_path = output_dir / "references.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    count = min(limit, len(dataset))
    records: list[dict[str, str]] = []

    for index in range(count):
        item = dataset[index]
        file_name = f"room_{index:03d}.jpg"
        image_path = image_dir / file_name

        if overwrite or not image_path.exists():
            item["images"].convert("RGB").save(image_path, quality=92)

        records.append(
            {
                "image": file_name,
                "reference": item["prompt"].strip(),
                "source": DATASET_ID,
            }
        )

    _write_jsonl(reference_path, records)
    return records


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
