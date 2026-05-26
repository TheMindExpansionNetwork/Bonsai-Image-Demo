#!/usr/bin/env python3
"""Smoke the three Sonic Forge generation legs by endpoint.

This is a client-side harness for a future warm service deployment. It does not
start models; it verifies the already-running endpoints and saves outputs.

Environment:
  SONIC_FORGE_TTS_URL    default http://127.0.0.1:8091/v1/audio/speech
  SONIC_FORGE_SFX_URL    optional, service-specific JSON endpoint
  SONIC_FORGE_IMAGE_URL  default http://127.0.0.1:8093/v1/images/generations
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

OUT = Path(os.environ.get("SONIC_FORGE_SMOKE_OUT", "outputs/sonic-forge-omni-smoke"))
TTS_URL = os.environ.get("SONIC_FORGE_TTS_URL", "http://127.0.0.1:8091/v1/audio/speech")
SFX_URL = os.environ.get("SONIC_FORGE_SFX_URL", "")
IMAGE_URL = os.environ.get("SONIC_FORGE_IMAGE_URL", "http://127.0.0.1:8093/v1/images/generations")


def post_json(url: str, payload: dict, timeout: int = 600) -> tuple[int, bytes, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator configured local/service URL
            return r.status, r.read(), r.headers.get("content-type", "")
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc).encode("utf-8"), "error/plain"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"timestamp": int(time.time()), "outputs": {}, "checks": {}}

    # Image leg: Bonsai bridge /v1/images/generations.
    status, body, ctype = post_json(IMAGE_URL, {
        "model": "bonsai-image-4b-ternary-gemlite",
        "prompt": "a cinematic Sonic Forge control room with a glowing bonsai image generator module, photorealistic, soft blue rim light",
        "size": "512x512",
        "steps": 4,
        "seed": 20260526,
        "response_format": "b64_json",
    })
    receipt["checks"]["image"] = {"url": IMAGE_URL, "status": status, "content_type": ctype}
    if status == 200:
        data = json.loads(body)
        png = base64.b64decode(data["data"][0]["b64_json"])
        path = OUT / "bonsai_image.png"
        path.write_bytes(png)
        receipt["outputs"]["image"] = str(path)

    # Voice leg: vLLM-Omni/OpenAI style audio speech. Payload may need voice/model
    # aliases adjusted per deployment.
    status, body, ctype = post_json(TTS_URL, {
        "model": "moss-tts-1.5",
        "voice": "sonic_forge_female",
        "input": "Okay, like, Sonic Forge is online. We have voice, sound effects, and now images in the same pipeline.",
        "response_format": "wav",
    })
    receipt["checks"]["tts"] = {"url": TTS_URL, "status": status, "content_type": ctype}
    if status == 200 and (body.startswith(b"RIFF") or "audio" in ctype):
        path = OUT / "moss_tts.wav"
        path.write_bytes(body)
        receipt["outputs"]["tts"] = str(path)

    # Optional SFX leg: service-specific until we standardize the MOSS SFX route.
    if SFX_URL:
        status, body, ctype = post_json(SFX_URL, {
            "prompt": "sparkly sci-fi interface chimes with soft studio ambience",
            "duration": 4.0,
            "response_format": "wav",
        })
        receipt["checks"]["sfx"] = {"url": SFX_URL, "status": status, "content_type": ctype}
        if status == 200 and (body.startswith(b"RIFF") or "audio" in ctype):
            path = OUT / "moss_sfx.wav"
            path.write_bytes(body)
            receipt["outputs"]["sfx"] = str(path)

    receipt_path = OUT / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
