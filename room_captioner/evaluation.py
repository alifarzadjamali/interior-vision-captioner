"""Simple, transparent checks for comparing caption runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

INTERIOR_TERMS = {
    "armchair",
    "bathroom",
    "bed",
    "bedroom",
    "cabinet",
    "chair",
    "couch",
    "curtain",
    "ceiling",
    "coffee",
    "desk",
    "dining",
    "door",
    "fireplace",
    "floor",
    "furniture",
    "kitchen",
    "lamp",
    "living",
    "mirror",
    "plant",
    "room",
    "rug",
    "shelf",
    "sofa",
    "table",
    "television",
    "wall",
    "window",
}


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep plain word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def score_caption(caption: str, reference: str) -> dict[str, float]:
    """Measure bag-of-words overlap and the amount of interior vocabulary."""
    predicted = Counter(tokenize(caption))
    expected = Counter(tokenize(reference))
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    details = len(
        {
            token[:-1] if token.endswith("s") else token
            for token in predicted
            if token in INTERIOR_TERMS
            or (token.endswith("s") and token[:-1] in INTERIOR_TERMS)
        }
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "interior_terms": float(details),
        "words": float(sum(predicted.values())),
    }


def evaluate_files(
    prediction_path: str | Path,
    reference_path: str | Path = "data/references.jsonl",
) -> dict[str, object]:
    """Match captions by image name and return per-image and average scores."""
    predictions = {item["image"]: item for item in read_jsonl(prediction_path)}
    references = {item["image"]: item for item in read_jsonl(reference_path)}
    shared_images = sorted(predictions.keys() & references.keys())
    if not shared_images:
        raise ValueError("No matching image names were found in the two files")

    rows = []
    for image_name in shared_images:
        prediction = predictions[image_name]
        reference = references[image_name]
        scores = score_caption(prediction["caption"], reference["reference"])
        rows.append(
            {
                "image": image_name,
                "caption": prediction["caption"],
                "reference": reference["reference"],
                **{key: round(value, 4) for key, value in scores.items()},
            }
        )

    metric_names = ("precision", "recall", "f1", "interior_terms", "words")
    averages = {
        name: round(sum(float(row[name]) for row in rows) / len(rows), 4)
        for name in metric_names
    }
    distinct_captions = {" ".join(tokenize(row["caption"])) for row in rows}
    averages["unique_caption_rate"] = round(len(distinct_captions) / len(rows), 4)
    return {"images": len(rows), "averages": averages, "results": rows}


def compare_files(
    baseline_path: str | Path,
    guided_path: str | Path,
    reference_path: str | Path = "data/references.jsonl",
) -> dict[str, object]:
    """Evaluate two runs and summarize the change in each average metric."""
    baseline = evaluate_files(baseline_path, reference_path)
    guided = evaluate_files(guided_path, reference_path)
    baseline_averages = baseline["averages"]
    guided_averages = guided["averages"]
    change = {
        key: round(float(guided_averages[key]) - float(value), 4)
        for key, value in baseline_averages.items()
    }
    return {"baseline": baseline_averages, "guided": guided_averages, "change": change}


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_report(report: dict[str, object], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
