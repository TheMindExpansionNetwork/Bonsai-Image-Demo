#!/usr/bin/env python3
"""OpenAI/vLLM-Omni compatible image endpoint bridge for Bonsai Image.

This does not load Bonsai weights itself. It expects the normal Bonsai studio
backend from `scripts/serve.sh` to be running, then exposes a small subset of
vLLM-Omni/OpenAI-compatible routes:

- GET  /health
- GET  /v1/models
- POST /v1/images/generations

The bridge forwards image requests to Bonsai's warm `/generate` route and
returns `b64_json`, matching the OpenAI images response shape. This lets a
Sonic Forge pipeline treat Bonsai like the image-generation leg next to MOSS
TTS 1.5 and MOSS SFX services.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BONSAI_BACKEND_URL = os.environ.get("BONSAI_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_MODEL = os.environ.get("BONSAI_OMNI_MODEL", "bonsai-image-4b-ternary-gemlite")
DEFAULT_STEPS = int(os.environ.get("BONSAI_OMNI_STEPS", "4"))
DEFAULT_SIZE = os.environ.get("BONSAI_OMNI_SIZE", "512x512")
REQUEST_TIMEOUT = float(os.environ.get("BONSAI_OMNI_TIMEOUT", "600"))

app = FastAPI(title="Bonsai Image vLLM-Omni Bridge", version="0.1.0")


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str | None = DEFAULT_MODEL
    n: int = Field(default=1, ge=1, le=1, description="Bonsai bridge currently supports one image per request")
    size: str = DEFAULT_SIZE
    seed: int | None = None
    steps: int | None = DEFAULT_STEPS
    output_format: str = Field(default="png", pattern="^(png)$")
    response_format: str = Field(default="b64_json", pattern="^(b64_json|url)$")


def _parse_size(size: str) -> tuple[int, int]:
    clean = size.lower().replace("×", "x")
    try:
        w_s, h_s = clean.split("x", 1)
        width, height = int(w_s), int(h_s)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"size must be WxH, got {size!r}") from exc
    if width % 16 or height % 16 or width < 256 or height < 256 or width > 2048 or height > 2048:
        raise HTTPException(status_code=400, detail="size must be 256..2048 per side and divisible by 16")
    return width, height


def _http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - local/operator-configured URL
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Bonsai backend HTTP {exc.code} at {url}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Bonsai backend unavailable at {url}: {exc}") from exc


def _post_bonsai_generate(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BONSAI_BACKEND_URL}/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "image/png"},
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:  # noqa: S310 - local/operator-configured URL
            content_type = r.headers.get("content-type", "")
            data = r.read()
            if "image" not in content_type and not data.startswith(b"\x89PNG"):
                raise HTTPException(status_code=502, detail=f"Bonsai returned non-image content-type={content_type!r}")
            return data
    except urllib.error.HTTPError as exc:
        err = exc.read(500).decode("utf-8", "replace")
        raise HTTPException(status_code=502, detail=f"Bonsai /generate HTTP {exc.code}: {err}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Bonsai /generate unavailable: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    backend = _http_json(f"{BONSAI_BACKEND_URL}/backends")
    return {"status": "ok", "bridge": "bonsai-vllm-omni", "bonsai_backend_url": BONSAI_BACKEND_URL, "backend": backend}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model", "owned_by": "PrismML-Eng/Bonsai-Image-Demo"},
            {"id": "bonsai-image-4b-binary-gemlite", "object": "model", "owned_by": "PrismML-Eng/Bonsai-Image-Demo"},
        ],
    }


@app.post("/v1/images/generations")
def image_generations(request: ImageGenerationRequest) -> dict[str, Any]:
    width, height = _parse_size(request.size)
    backend = "bonsai-binary-gemlite" if request.model and "binary" in request.model else "bonsai-ternary-gemlite"
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "steps": request.steps or DEFAULT_STEPS,
        "height": height,
        "width": width,
        "backend": backend,
    }
    if request.seed is not None:
        payload["seed"] = int(request.seed)
    png = _post_bonsai_generate(payload)
    b64 = base64.b64encode(png).decode("ascii")
    # OpenAI permits b64_json or URL. Returning a data URL for url mode keeps
    # this bridge single-file and storage-free.
    image_obj = {"b64_json": b64} if request.response_format == "b64_json" else {"url": f"data:image/png;base64,{b64}"}
    image_obj.update({"revised_prompt": request.prompt, "size": request.size, "output_format": "png"})
    return {"created": int(time.time()), "data": [image_obj]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bonsai vLLM-Omni image bridge")
    parser.add_argument("--host", default=os.environ.get("BONSAI_OMNI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BONSAI_OMNI_PORT", "8093")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
