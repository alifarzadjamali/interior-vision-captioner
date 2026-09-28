import json

from PIL import Image

from room_captioner.pipeline import caption_folder, find_images


class StubCaptioner:
    model_id = "test-model"

    def caption(self, image_path, prompt=None, max_new_tokens=40):
        return f"{prompt or 'plain'} room"


def test_caption_folder_writes_reproducible_jsonl(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "b.png")
    Image.new("RGB", (8, 8), "white").save(image_dir / "a.jpg")
    (image_dir / "notes.txt").write_text("skip me")
    output_path = tmp_path / "captions.jsonl"

    records = caption_folder(StubCaptioner(), image_dir, output_path, mode="guided")

    assert [path.name for path in find_images(image_dir)] == ["a.jpg", "b.png"]
    assert [record["image"] for record in records] == ["a.jpg", "b.png"]
    saved = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert saved == records
