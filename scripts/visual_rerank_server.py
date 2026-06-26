"""FastAPI-сервис визуального реранкера Qwen3-VL-Reranker-2B.

Cross-encoder: вход (query, document) где document — текст и/или картинка →
relevance score (сигмоид 0..1, выше = релевантнее). Модель грузится один раз
на GPU (CUDA_VISIBLE_DEVICES задаётся снаружи, в проде GPU2).

ПОЧЕМУ НЕ vLLM: vLLM-путь для этой модели багован (vllm#35412 даёт реверсивные
скоры), плюс vLLM 0.22/0.23 не знают архитектуру Qwen3VLForSequenceClassification.
Поэтому инференс — напрямую через transformers, по референсной реализации из
карточки модели (`scripts/qwen3_vl_reranker.py`).

Логика скоринга (как в карточке):
  • грузим Qwen3VLForConditionalGeneration, берём backbone `.model`;
  • из lm_head собираем бинарный линейный слой w = w[yes] - w[no];
  • score = sigmoid( w · last_hidden_state[:, -1] ).

Запуск (см. deploy/vllm-visual-reranker.service):
  CUDA_VISIBLE_DEVICES=2 uvicorn visual_rerank_server:app --host 127.0.0.1 --port 8009

Endpoint:
  POST /rerank {query: str, documents: [{text?: str, image_b64?: str}]}
    → {scores: [float]}   # сигмоид 0..1, порядок == порядку documents
  GET  /healthz → {status, model_loaded}
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visual_rerank")

MODEL_PATH = os.environ.get("VISUAL_RERANK_MODEL_PATH", "/root/models/Qwen3-VL-Reranker-2B")
MAX_LENGTH = int(os.environ.get("VISUAL_RERANK_MAX_LENGTH", "8192"))

# Пиксельный бюджет картинки (из карточки модели).
IMAGE_FACTOR = 16 * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR

DEFAULT_INSTRUCTION = "Retrieve text relevant to the user's query."
SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)


class Qwen3VLReranker:
    """Референсная реализация скоринга из карточки модели (self-contained)."""

    def __init__(self, model_name_or_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device)
        self.model = lm.model  # backbone → last_hidden_state
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True, padding_side="left"
        )

        vocab = self.processor.tokenizer.get_vocab()
        token_yes, token_no = vocab["yes"], vocab["no"]
        self.score_linear = self._binary_linear(lm, token_yes, token_no)
        self.score_linear.eval().to(self.device).to(self.model.dtype)

    @staticmethod
    def _binary_linear(model: Any, token_yes: int, token_no: int) -> torch.nn.Linear:
        w = model.lm_head.weight.data
        d = w[token_yes].size(0)
        layer = torch.nn.Linear(d, 1, bias=False)
        with torch.no_grad():
            layer.weight[0] = w[token_yes] - w[token_no]
        return layer

    @staticmethod
    def _content(text: str | None, image: Image.Image | None, prefix: str) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prefix}]
        if not text and image is None:
            content.append({"type": "text", "text": "NULL"})
            return content
        if image is not None:
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            )
        if text:
            content.append({"type": "text", "text": text})
        return content

    def _build_messages(
        self,
        query_text: str | None,
        doc_text: str | None,
        doc_image: Image.Image | None,
        instruction: str,
    ) -> list[dict]:
        user_content: list[dict] = [{"type": "text", "text": "<Instruct>: " + instruction}]
        user_content.extend(self._content(query_text, None, "<Query>:"))
        user_content.extend(self._content(doc_text, doc_image, "\n<Document>:"))
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    def _tokenize(self, messages: list[dict]) -> dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("process_vision_info failed: %s", exc)
            images, videos, video_kwargs = None, None, {}
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            truncation=False,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs.to(self.model.device)

    @torch.no_grad()
    def score(
        self,
        query_text: str | None,
        doc_text: str | None,
        doc_image: Image.Image | None,
        instruction: str,
    ) -> float:
        messages = self._build_messages(query_text, doc_text, doc_image, instruction)
        inputs = self._tokenize(messages)
        last = self.model(**inputs).last_hidden_state[:, -1]
        logit = self.score_linear(last)
        return float(torch.sigmoid(logit).squeeze(-1).item())


# --- FastAPI ---------------------------------------------------------------


class Document(BaseModel):
    text: str | None = None
    image_b64: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    instruction: str | None = None


class RerankResponse(BaseModel):
    scores: list[float]


app = FastAPI(title="Qwen3-VL-Reranker-2B")
_model: Qwen3VLReranker | None = None


def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.on_event("startup")
def _load() -> None:
    global _model
    logger.info("Loading reranker from %s ...", MODEL_PATH)
    _model = Qwen3VLReranker(MODEL_PATH)
    logger.info("Reranker loaded on %s", _model.device)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    instruction = req.instruction or DEFAULT_INSTRUCTION
    scores: list[float] = []
    for doc in req.documents:
        image = _decode_image(doc.image_b64) if doc.image_b64 else None
        scores.append(_model.score(req.query, doc.text, image, instruction))
    return RerankResponse(scores=scores)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8009)
