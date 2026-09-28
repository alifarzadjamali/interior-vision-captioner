"""Run captioning over a folder and save tidy JSONL results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CAPTION_MODES = {
    "baseline": None,
    "guided": "a living room with",
}


class Captioner(Protocol):
    model_id: str

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_new_tokens: int = 40,
    ) -> str: ...


def find_images(image_dir: str | Path) -> list[Path]:
    """Return supported images in a stable order."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def caption_folder(
    captioner: Captioner,
    image_dir: str | Path,
    output_path: str | Path,
    mode: str = "guided",
    max_new_tokens: int = 40,
    custom_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Caption every image in a folder and write one JSON object per line."""
    if mode not in CAPTION_MODES:
        choices = ", ".join(CAPTION_MODES)
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {choices}")

    image_paths = find_images(image_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {image_dir}")

    prompt = custom_prompt if custom_prompt is not None else CAPTION_MODES[mode]
    records = []
    for image_path in image_paths:
        caption = captioner.caption(image_path, prompt, max_new_tokens)
        records.append(
            {
                "image": image_path.name,
                "caption": caption,
                "mode": mode,
                "prompt": prompt or "",
                "model": captioner.model_id,
            }
        )
        print(f"{image_path.name}: {caption}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return records
