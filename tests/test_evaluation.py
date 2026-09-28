import json

from room_captioner.evaluation import evaluate_files, score_caption, tokenize


def test_tokenize_normalizes_punctuation_and_case():
    assert tokenize("A Bright, BLUE room!") == ["a", "bright", "blue", "room"]


def test_score_caption_counts_word_overlap_and_interior_terms():
    scores = score_caption(
        "a living room with blue sofas and chairs",
        "a living room with sofas and chairs",
    )

    assert scores["precision"] == 7 / 8
    assert scores["recall"] == 1.0
    assert scores["interior_terms"] == 4.0


def test_evaluate_files_reports_caption_diversity(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    references = tmp_path / "references.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"image": name, "caption": "the same room"})
            for name in ("one.jpg", "two.jpg")
        )
    )
    references.write_text(
        "\n".join(
            json.dumps({"image": name, "reference": "a room"})
            for name in ("one.jpg", "two.jpg")
        )
    )

    report = evaluate_files(predictions, references)

    assert report["averages"]["unique_caption_rate"] == 0.5
