# Sonic Forge Omni Bonsai setup receipt — 20260526T200922Z

## Repos

- Upstream: https://github.com/PrismML-Eng/Bonsai-Image-Demo
- Fork origin: https://github.com/TheMindExpansionNetwork/Bonsai-Image-Demo.git
- Local clone: `/opt/data/workspace/github-forks/Bonsai-image-demo`
- Branch during setup: `main`
- Upstream base HEAD: `cc1ac6b482c2be4e2e9950d94901903d969b1b8b`

## Setup performed

- Forked/cloned Bonsai Image Demo under `TheMindExpansionNetwork`.
- Added `upstream` remote pointing to `PrismML-Eng/Bonsai-Image-Demo`.
- Ran local dependency setup with `SKIP_DOWNLOAD=1 ./setup.sh`.
- Added `scripts/vllm_omni_bonsai_bridge.py` to expose Bonsai's native `/generate` route as an OpenAI/vLLM-Omni-style `/v1/images/generations` endpoint.
- Added `scripts/sonic_forge_omni_smoke.py` as a client smoke harness for MOSS TTS + optional MOSS SFX + Bonsai image endpoints.
- Added `docs/sonic-forge-omni-pipeline.md` with operator commands and endpoint map.

## Verification

- Python compile check: passed for bridge and smoke scripts.
- Dependency setup: passed with `SKIP_DOWNLOAD=1`.
- Local GPU: no NVIDIA GPU visible via `nvidia-smi`; generation cannot run on this host.
- Bonsai model download: attempted with the configured token path, but stopped because local disk did not have enough free space for the gated weights/cache. Partial model download was removed.
- Current disk after cleanup: `/dev/sda1       193G  188G  5.7G  98% /opt/data`

## License / release caution

Upstream repository metadata currently reports no explicit license. Keep this fork/integration private until PrismML publishes or clarifies licensing.

## Next GPU-host command

On a GPU host with enough disk and Bonsai/HF token access:

```bash
BONSAI_TOKEN='<token-from-secret-store>' ./scripts/download_model.sh ternary
BONSAI_VARIANT=ternary ./scripts/serve.sh
BONSAI_BACKEND_URL=http://127.0.0.1:8000 ./.venv/bin/python scripts/vllm_omni_bonsai_bridge.py --port 8093
```
