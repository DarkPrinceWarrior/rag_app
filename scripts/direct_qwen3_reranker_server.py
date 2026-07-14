"""Deterministic OpenAI-compatible scorer for Qwen3-Reranker-4B.

The application client already applies Qwen's official prompt prefix/suffix.
This service therefore concatenates each API query/document pair verbatim and
scores the final token with the official ``no``/``yes`` softmax recipe.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("direct_qwen3_reranker")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("DIRECT_RERANK_MODEL_PATH", "/root/models/Qwen3-Reranker-4B")
SERVED_MODEL_NAME = os.environ.get("DIRECT_RERANK_SERVED_MODEL_NAME", "qwen3-reranker-4b")
MAX_LENGTH = int(os.environ.get("DIRECT_RERANK_MAX_LENGTH", "8192"))
MICRO_BATCH_SIZE = int(os.environ.get("DIRECT_RERANK_BATCH_SIZE", "4"))
_QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _pack_token_ids(
    token_ids: list[int],
    suffix_token_ids: tuple[int, ...],
    *,
    max_length: int,
) -> list[int]:
    if not suffix_token_ids or len(suffix_token_ids) >= max_length:
        raise RuntimeError("reranker suffix does not fit the model context")
    suffix_length = len(suffix_token_ids)
    if tuple(token_ids[-suffix_length:]) != suffix_token_ids:
        raise RuntimeError("reranker input is missing the official suffix")
    body = token_ids[:-suffix_length]
    return [*body[: max_length - suffix_length], *suffix_token_ids]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


DocumentText = Annotated[str, Field(min_length=1, max_length=5000)]


class RerankRequest(_StrictModel):
    model: str = Field(min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=4096)
    documents: list[DocumentText] = Field(min_length=1, max_length=128)


class RerankResult(_StrictModel):
    index: int = Field(ge=0)
    relevance_score: float = Field(ge=0.0, le=1.0)


class RerankResponse(_StrictModel):
    model: str
    results: list[RerankResult]


class HealthResponse(_StrictModel):
    status: Literal["ok", "loading"]
    model: str
    model_loaded: bool
    deterministic_algorithms: bool
    max_length: int
    micro_batch_size: int


class ModelCard(_StrictModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: Literal["local"] = "local"


class ModelList(_StrictModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class Qwen3RerankerRuntime:
    def __init__(self, model_path: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3 reranker requires CUDA")
        if MAX_LENGTH < 1 or MICRO_BATCH_SIZE < 1:
            raise RuntimeError("reranker length and batch size must be positive")

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
        )
        vocabulary = self._tokenizer.get_vocab()
        self._token_no = vocabulary["no"]
        self._token_yes = vocabulary["yes"]
        self._suffix_token_ids = tuple(
            self._tokenizer(
                _QWEN3_RERANK_SUFFIX,
                add_special_tokens=False,
            )["input_ids"]
        )
        if not self._suffix_token_ids or len(self._suffix_token_ids) >= MAX_LENGTH:
            raise RuntimeError("official reranker suffix is invalid")
        if self._tokenizer.pad_token_id is None:
            raise RuntimeError("reranker tokenizer has no padding token")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda()
        self._model.eval()
        self._lock = threading.Lock()

    @contextmanager
    def _serialized_inference(self) -> Iterator[None]:
        with self._lock, torch.inference_mode():
            yield

    def score(self, query: str, documents: list[str]) -> list[float]:
        packed_inputs = [
            _pack_token_ids(
                list(
                    self._tokenizer(
                        query + document,
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"]
                ),
                self._suffix_token_ids,
                max_length=MAX_LENGTH,
            )
            for document in documents
        ]
        scores: list[float] = []
        with self._serialized_inference():
            for offset in range(0, len(packed_inputs), MICRO_BATCH_SIZE):
                batch = packed_inputs[offset : offset + MICRO_BATCH_SIZE]
                padded_length = max(len(item) for item in batch)
                input_ids = torch.full(
                    (len(batch), padded_length),
                    self._tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self._model.device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for index, item in enumerate(batch):
                    length = len(item)
                    input_ids[index, -length:] = torch.tensor(
                        item,
                        dtype=torch.long,
                        device=self._model.device,
                    )
                    attention_mask[index, -length:] = 1
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, [self._token_no, self._token_yes]]
                probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
                scores.extend(float(value) for value in probabilities.cpu().tolist())
        if len(scores) != len(documents) or not all(
            math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores
        ):
            raise RuntimeError("reranker returned invalid scores")
        return scores


_runtime: Qwen3RerankerRuntime | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _runtime
    logger.info("Loading deterministic reranker from %s", MODEL_PATH)
    _runtime = Qwen3RerankerRuntime(MODEL_PATH)
    logger.info("Deterministic reranker loaded")
    try:
        yield
    finally:
        _runtime = None


app = FastAPI(title="Qwen3-Reranker-4B deterministic runtime", lifespan=_lifespan)


@app.get("/healthz")
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok" if _runtime is not None else "loading",
        model=SERVED_MODEL_NAME,
        model_loaded=_runtime is not None,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        max_length=MAX_LENGTH,
        micro_batch_size=MICRO_BATCH_SIZE,
    )


@app.get("/v1/models")
def models() -> ModelList:
    return ModelList(data=[ModelCard(id=SERVED_MODEL_NAME)])


@app.post("/v1/rerank")
def rerank(request: RerankRequest) -> RerankResponse:
    if _runtime is None:
        raise HTTPException(status_code=503, detail="reranker is not loaded")
    if request.model != SERVED_MODEL_NAME:
        raise HTTPException(status_code=404, detail="requested model is not served")
    scores = _runtime.score(request.query, request.documents)
    results = sorted(
        (
            RerankResult(index=index, relevance_score=score)
            for index, score in enumerate(scores)
        ),
        key=lambda item: (-item.relevance_score, item.index),
    )
    return RerankResponse(model=SERVED_MODEL_NAME, results=results)
