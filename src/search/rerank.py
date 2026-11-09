"""Optional Qwen2.5-VL top-3 reranker.

The reranker asks a yes/no question for each candidate image:

    "Does this image show: '<query>'? Answer yes or no."

It is deliberately separate from the retrieval index: loading a 3B VLM costs
seconds, so it is only used on the top 3 candidates and its latency is reported
rather than hidden. ``scripts/measure_rerank.py`` measures that latency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from src.config import Config
from src.index.frames import resolve_device
from src.utils.logging import get_logger

_LOG = get_logger("search.rerank")

DEFAULT_RERANK_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenReranker:
    """Lazy-loaded Qwen2.5-VL yes/no reranker."""

    def __init__(self, config: Config) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_name = str(
            config.get("captions.qwen_model") or DEFAULT_RERANK_MODEL
        )
        self.device = resolve_device(config.get("captions.device", "auto"))
        _LOG.info("loading reranker %s on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    def yes_no(self, query: str, image_path: str | Path) -> tuple[bool, str]:
        import torch

        image = Image.open(image_path).convert("RGB")
        prompt = (
            f"Does this image show: '{query}'? "
            "Answer yes or no in one word."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=4, do_sample=False
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        answer = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        return answer.lower().startswith("yes"), answer

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        *,
        top_n: int = 3,
    ) -> tuple[list[dict[str, Any]], float, list[bool | None]]:
        """Return reordered candidates, elapsed milliseconds, and yes/no flags."""
        started = time.perf_counter()
        selected = list(candidates)[: max(1, int(top_n))]
        flags: list[bool | None] = []
        for candidate in selected:
            path = (
                candidate.get("peak_path")
                or candidate.get("path")
                or candidate.get("thumbnail")
            )
            if not path:
                flags.append(None)
                continue
            accepted, _answer = self.yes_no(query, path)
            flags.append(accepted)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        accepted = [
            candidate
            for candidate, flag in zip(selected, flags)
            if flag is True
        ]
        rejected = [
            candidate
            for candidate, flag in zip(selected, flags)
            if flag is not True
        ]
        return accepted + rejected, elapsed_ms, flags
