"""FastAPI service for a quantized MobileNetV2 TFLite model."""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow import lite

        Interpreter = lite.Interpreter
    except ImportError:
        Interpreter = None


LOGGER = logging.getLogger("edge_inference")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/mobilenet_v2_int8.tflite"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))


@dataclass
class Runtime:
    interpreter: Any = None
    input_detail: dict | None = None
    output_detail: dict | None = None
    error: str | None = None


runtime = Runtime()
invoke_lock = threading.Lock()


def load_model(path: Path = MODEL_PATH) -> None:
    runtime.interpreter = None
    runtime.input_detail = None
    runtime.output_detail = None
    runtime.error = None
    if Interpreter is None:
        runtime.error = "TensorFlow Lite runtime is not installed"
        return
    if not path.is_file():
        runtime.error = f"Model not found: {path}"
        return
    try:
        interpreter = Interpreter(model_path=str(path), num_threads=max(1, int(os.getenv("TFLITE_THREADS", "1"))))
        interpreter.allocate_tensors()
        runtime.interpreter = interpreter
        runtime.input_detail = interpreter.get_input_details()[0]
        runtime.output_detail = interpreter.get_output_details()[0]
    except Exception as exc:
        runtime.error = f"Model load failed: {exc}"
        LOGGER.exception(runtime.error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="MobileNetV2 Edge Inference",
    description="A reproducible TFLite inference service.",
    version="2.0.0",
    lifespan=lifespan,
)


def _quantize(array: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype not in (np.int8, np.uint8):
        return array.astype(dtype)
    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise HTTPException(status_code=500, detail="Invalid model input quantization metadata")
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(array / scale + zero_point), bounds.min, bounds.max).astype(dtype)


def preprocess_image(content: bytes, detail: dict) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed configured limit")
            image = source.convert("RGB")
            height, width = (int(detail["shape"][1]), int(detail["shape"][2]))
            resize_short_side = round(max(height, width) * 256 / 224)
            scale = resize_short_side / min(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.BILINEAR,
            )
            left = (image.width - width) // 2
            top = (image.height - height) // 2
            image = image.crop((left, top, left + width, top + height))
            array = np.asarray(image, dtype=np.float32)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc
    normalized = (array / 127.5) - 1.0
    return _quantize(np.expand_dims(normalized, 0), detail)


def _require_model() -> None:
    if runtime.interpreter is None:
        raise HTTPException(status_code=503, detail=runtime.error or "Model unavailable")


@app.get("/")
def root() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "model_loaded": runtime.interpreter is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/health")
def health() -> dict:
    _require_model()
    return {
        "status": "healthy",
        "model_path": str(MODEL_PATH),
        "rss_mib": round(psutil.Process().memory_info().rss / 2**20, 2),
    }


@app.get("/metadata")
def metadata() -> dict:
    _require_model()
    return {
        "input": {
            "shape": runtime.input_detail["shape"].tolist(),
            "dtype": str(runtime.input_detail["dtype"]),
            "quantization": list(runtime.input_detail["quantization"]),
        },
        "output": {
            "shape": runtime.output_detail["shape"].tolist(),
            "dtype": str(runtime.output_detail["dtype"]),
            "quantization": list(runtime.output_detail["quantization"]),
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    _require_model()
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds configured limit")

    input_data = preprocess_image(content, runtime.input_detail)
    started = time.perf_counter_ns()
    with invoke_lock:
        runtime.interpreter.set_tensor(runtime.input_detail["index"], input_data)
        runtime.interpreter.invoke()
        scores = runtime.interpreter.get_tensor(runtime.output_detail["index"])[0]
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000

    scale, zero_point = runtime.output_detail["quantization"]
    if scale:
        scores = (scores.astype(np.float32) - zero_point) * scale
    top_indices = np.argsort(scores)[-5:][::-1]
    return {
        "predictions": [
            {"class_index": int(index), "score": round(float(scores[index]), 6)}
            for index in top_indices
        ],
        "inference_ms": round(latency_ms, 3),
    }
