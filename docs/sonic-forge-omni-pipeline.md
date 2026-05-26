# Sonic Forge Omni pipeline: MOSS TTS 1.5 + MOSS SFX + Bonsai Image

This fork adds a small OpenAI/vLLM-Omni-compatible bridge in front of the Bonsai Image demo so the image generator can sit beside the existing Sonic Forge audio lanes:

- **Voice:** MOSS TTS 1.5 / MOSS-TTS-Realtime fine-tuned voice endpoint.
- **SFX:** MOSS SoundEffect v2 endpoint or offline generator.
- **Image:** Bonsai Image 4B via this repo's warm Bonsai studio backend, exposed as `/v1/images/generations` by `scripts/vllm_omni_bonsai_bridge.py`.

The bridge is intentionally thin: it does not pretend Bonsai is loaded inside vLLM-Omni. It adapts Bonsai's native `/generate` route to the same API shape we use for vLLM-Omni image calls. That gives us one operator/client interface while keeping Bonsai's gemlite/HQQ runtime intact.

## Local setup status

A local dependency setup can be performed with:

```bash
SKIP_DOWNLOAD=1 ./setup.sh
```

Model download still requires:

- enough disk for Bonsai weights and caches;
- a Hugging Face/Bonsai token while the upstream weights are gated;
- an NVIDIA GPU for Linux generation.

This machine currently has no visible NVIDIA GPU, so generation should run on Modal/RunPod/a GPU VPS or a different CUDA host. The local repo is still useful for packaging, API bridge work, and client integration.

## Start Bonsai backend

After the model is downloaded on a GPU host:

```bash
# Ternary is recommended quality/default lane.
BONSAI_VARIANT=ternary ./scripts/serve.sh

# Optional custom ports:
BACKEND_PORT=8800 FRONTEND_PORT=3100 BONSAI_VARIANT=ternary ./scripts/serve.sh
```

The native Bonsai backend exposes:

- `GET /backends`
- `POST /generate`

## Start the vLLM-Omni-compatible image bridge

In another shell:

```bash
# Default: forwards to http://127.0.0.1:8000 and listens on 127.0.0.1:8093
./.venv/bin/python scripts/vllm_omni_bonsai_bridge.py

# Custom Bonsai backend / bridge port:
BONSAI_BACKEND_URL=http://127.0.0.1:8800 \
BONSAI_OMNI_PORT=8093 \
./.venv/bin/python scripts/vllm_omni_bonsai_bridge.py
```

Health/model probes:

```bash
curl -s http://127.0.0.1:8093/health | jq .
curl -s http://127.0.0.1:8093/v1/models | jq .
```

Image request in OpenAI/vLLM-Omni style:

```bash
curl -s http://127.0.0.1:8093/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bonsai-image-4b-ternary-gemlite",
    "prompt": "a cinematic photo of a tiny glowing bonsai tree on a spaceship mixing console, soft fog, studio lighting",
    "size": "512x512",
    "seed": 9909,
    "steps": 4,
    "response_format": "b64_json"
  }' \
  | jq -r '.data[0].b64_json' | base64 -d > outputs/sonic_forge_bonsai_probe.png
```

## Suggested Sonic Forge service map

Use separate warm services first; combine behind a higher-level orchestrator after each leg is verified.

```text
Sonic Forge client/orchestrator
├── voice:  http://<moss-tts-host>:8091/v1/audio/speech
├── sfx:    http://<moss-sfx-host>:8092/generate or /v1/audio/effects
└── image:  http://<bonsai-host>:8093/v1/images/generations
```

Example environment contract:

```bash
export SONIC_FORGE_TTS_URL=http://127.0.0.1:8091/v1/audio/speech
export SONIC_FORGE_SFX_URL=http://127.0.0.1:8092/generate
export SONIC_FORGE_IMAGE_URL=http://127.0.0.1:8093/v1/images/generations
```

## Guardrails

- Keep Bonsai model weights out of Git; `models/` is ignored.
- Keep `BONSAI_TOKEN`, `HF_TOKEN`, and any Modal/RunPod credentials in environment/secret stores only.
- Label generated media honestly: MOSS TTS/SFX/Bonsai generated output, not live capture.
- Treat the Bonsai upstream as currently **no explicit repo license** until PrismML publishes or clarifies licensing. Keep derivative integration private unless licensing is resolved.
