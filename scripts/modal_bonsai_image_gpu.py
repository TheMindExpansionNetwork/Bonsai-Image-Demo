"""Modal GPU runner for Bonsai Image Demo.

This is intentionally separate from the local bridge/server code so a CPU smoke
or local edit does not build a heavy CUDA image. It clones the GitHub fork inside
Modal, uses persistent Volumes for Hugging Face/model/output caches, and returns
only JSON-serializable receipts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "sonic-forge-bonsai-image-gpu"
REPO_URL = "https://github.com/TheMindExpansionNetwork/Bonsai-Image-Demo.git"
BRANCH = "sonic-forge/omni-bonsai-bridge"
DEFAULT_PROMPT = "cinematic photo of a tiny bonsai synthesizer tree glowing in a cyberpunk music studio, realistic, detailed"
REPO_PATH = Path("/root/Bonsai-Image-Demo")
HF_HOME = Path("/cache/huggingface")
OUTPUT_ROOT = Path("/outputs/bonsai-image")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("bonsai-image-hf-cache", create_if_missing=True)
bonsai_models = modal.Volume.from_name("bonsai-image-models", create_if_missing=True)
bonsai_outputs = modal.Volume.from_name("bonsai-image-outputs", create_if_missing=True)

base_env = {
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": str(HF_HOME),
    "HUGGINGFACE_HUB_CACHE": str(HF_HOME / "hub"),
    "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
    "UV_LINK_MODE": "copy",
    # setup/download_model installs hf-transfer; keep the fast downloader enabled.
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "build-essential", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "export PATH=/root/.local/bin:$PATH && git clone --branch sonic-forge/omni-bonsai-bridge https://github.com/TheMindExpansionNetwork/Bonsai-Image-Demo.git /root/Bonsai-Image-Demo",
        "export PATH=/root/.local/bin:$PATH && cd /root/Bonsai-Image-Demo && SKIP_DOWNLOAD=1 BONSAI_PACKAGE_MIN_AGE_DAYS=0 ./setup.sh",
    )
    .env(base_env)
)


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=timeout,
    )


def _du(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception as exc:  # JSON-safe diagnostics only
        return f"error:{type(exc).__name__}:{exc}"


def _normalize_hf_token_env() -> dict[str, str]:
    env = os.environ.copy()
    token = (
        env.get("BONSAI_TOKEN")
        or env.get("HF_TOKEN")
        or env.get("HUGGING_FACE_HUB_TOKEN")
        or env.get("HUGGINGFACE_HUB_TOKEN")
        or env.get("HUGGINGFACE_TOKEN")
    )
    if token:
        env["BONSAI_TOKEN"] = token
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    return env


def _ensure_models_symlink() -> None:
    target = Path("/models")
    target.mkdir(parents=True, exist_ok=True)
    repo_models = REPO_PATH / "models"
    if repo_models.is_symlink() or repo_models.exists():
        if repo_models.is_symlink() and repo_models.resolve() == target:
            return
        if repo_models.is_dir() and not any(repo_models.iterdir()):
            repo_models.rmdir()
        else:
            # Keep an existing directory but prefer the volume by replacing only
            # when safe. This should not trigger in the build image.
            return
    repo_models.symlink_to(target, target_is_directory=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": hf_cache, "/models": bonsai_models, "/outputs": bonsai_outputs},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("hf-token")],
    timeout=60 * 60 * 2,
)
def generate_smoke(
    prompt: str = DEFAULT_PROMPT,
    model: str = "binary-gemlite",
    size: str = "512x512",
    steps: int = 4,
    seed: int = 2045,
) -> dict:
    """Download/sync a Bonsai model into a Modal Volume, then render one PNG."""
    t0 = time.time()
    _ensure_models_symlink()
    env = _normalize_hf_token_env()
    if not env.get("BONSAI_TOKEN"):
        raise RuntimeError("Missing HF/BONSAI token in Modal secrets; expected BONSAI_TOKEN or HF_TOKEN-like env var")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"bonsai_{model}_{size}_seed{seed}.png"

    download = _run(["./scripts/download_model.sh", "--model", model], cwd=REPO_PATH, env=env, timeout=60 * 60)
    render = _run(
        [
            "./scripts/generate.sh",
            "--model", model,
            "--prompt", prompt,
            "--size", size,
            "--steps", str(steps),
            "--seed", str(seed),
            "--output", str(output),
            "--force-gpu-run",
        ],
        cwd=REPO_PATH,
        env=env,
        timeout=60 * 60,
    )

    receipt = {
        "action": "generate_smoke",
        "repo_url": REPO_URL,
        "branch": BRANCH,
        "model": model,
        "prompt": prompt,
        "size": size,
        "steps": steps,
        "seed": seed,
        "output": str(output),
        "output_exists": output.exists(),
        "output_bytes": output.stat().st_size if output.exists() else 0,
        "nonempty_png": bool(output.exists() and output.stat().st_size > 1000 and output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"),
        "model_volume_du": _du(Path("/models")),
        "output_volume_du": _du(OUTPUT_ROOT),
        "hf_cache_du": _du(HF_HOME),
        "elapsed_sec": round(time.time() - t0, 2),
        "download_tail": download.stdout[-3000:],
        "render_tail": render.stdout[-3000:],
    }
    (OUTPUT_ROOT / f"receipt_{model}_{size}_seed{seed}.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    hf_cache.commit(); bonsai_models.commit(); bonsai_outputs.commit()
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


@app.function(image=image, volumes={"/cache": hf_cache, "/models": bonsai_models, "/outputs": bonsai_outputs}, timeout=60 * 10)
def report() -> dict:
    obj = {
        "action": "report",
        "repo_path": str(REPO_PATH),
        "models_du": _du(Path("/models")),
        "outputs_du": _du(Path("/outputs")),
        "hf_cache_du": _du(HF_HOME),
        "outputs": sorted(str(p.relative_to('/outputs')) for p in Path('/outputs').rglob('*') if p.is_file())[:200] if Path('/outputs').exists() else [],
    }
    print(json.dumps(obj, indent=2), flush=True)
    return obj


@app.local_entrypoint()
def main(action: str = "generate", prompt: str = "", model: str = "binary-gemlite", size: str = "512x512", steps: int = 4, seed: int = 2045):
    if action == "generate":
        print(generate_smoke.remote(prompt=prompt or DEFAULT_PROMPT, model=model, size=size, steps=steps, seed=seed))
    elif action == "report":
        print(report.remote())
    else:
        raise ValueError(action)
