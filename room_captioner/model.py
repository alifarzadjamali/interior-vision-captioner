"""Model loading and caption generation."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = "Salesforce/blip-image-captioning-base"


class BlipCaptioner:
    """A reusable BLIP captioner that loads the model only once."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()
        self.model_id = model_id

    def caption(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        max_new_tokens: int = 40,
    ) -> str:
        """Create one caption, optionally beginning it with a short prompt."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

        with torch.inference_mode():
            token_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                repetition_penalty=1.2,
            )

        return self.processor.decode(token_ids[0], skip_special_tokens=True).strip()
