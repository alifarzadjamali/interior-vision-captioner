"""Commands for downloading data, captioning rooms, and checking results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from room_captioner.data import DATASET_ID, download_dataset
from room_captioner.evaluation import compare_files, evaluate_files, save_report
from room_captioner.pipeline import CAPTION_MODES

DEFAULT_IMAGE_DIR = Path("data/images")
DEFAULT_OUTPUT_DIR = Path("data/outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="room-captioner",
        description="Caption interior images and compare prompt strategies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download the public sample dataset")
    download.add_argument("--limit", type=int, default=12)
    download.add_argument("--output-dir", type=Path, default=Path("data"))
    download.add_argument("--overwrite", action="store_true")

    caption = commands.add_parser("caption", help="Caption every image in a folder")
    caption.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    caption.add_argument("--mode", choices=CAPTION_MODES, default="guided")
    caption.add_argument(
        "--prompt",
        help="Override the selected mode with your own caption opening",
    )
    caption.add_argument("--output", type=Path)
    caption.add_argument("--max-new-tokens", type=int, default=40)
    caption.add_argument("--device", choices=("cpu", "cuda"))

    evaluate = commands.add_parser("evaluate", help="Score one caption file")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument(
        "--references", type=Path, default=Path("data/references.jsonl")
    )
    evaluate.add_argument("--output", type=Path)

    compare = commands.add_parser("compare", help="Compare baseline and guided runs")
    compare.add_argument(
        "--baseline", type=Path, default=DEFAULT_OUTPUT_DIR / "baseline.jsonl"
    )
    compare.add_argument(
        "--guided", type=Path, default=DEFAULT_OUTPUT_DIR / "guided.jsonl"
    )
    compare.add_argument(
        "--references", type=Path, default=Path("data/references.jsonl")
    )
    compare.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / "comparison.json"
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "download":
        records = download_dataset(args.output_dir, args.limit, args.overwrite)
        print(f"Saved {len(records)} rooms from {DATASET_ID} to {args.output_dir}")
        return

    if args.command == "caption":
        from room_captioner.model import BlipCaptioner
        from room_captioner.pipeline import caption_folder

        output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.mode}.jsonl"
        captioner = BlipCaptioner(device=args.device)
        records = caption_folder(
            captioner,
            args.images,
            output_path,
            args.mode,
            args.max_new_tokens,
            args.prompt,
        )
        print(f"Saved {len(records)} captions to {output_path}")
        return

    if args.command == "evaluate":
        report = evaluate_files(args.predictions, args.references)
        if args.output:
            save_report(report, args.output)
            print(f"Saved report to {args.output}")
        print(json.dumps(report["averages"], indent=2))
        return

    report = compare_files(args.baseline, args.guided, args.references)
    save_report(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"Saved comparison to {args.output}")
