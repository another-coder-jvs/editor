#!/usr/bin/env python3
"""
Model Download Script — AI Image Editor
========================================
Downloads ALL required models in one go so the app can run fully offline.

Usage:
    python download_models.py              # download everything
    python download_models.py --list       # list models without downloading
    python download_models.py --only sam2  # download a specific model (repeatable)
    python download_models.py --skip flux  # skip specific models (repeatable)
    python download_models.py --weights-dir /path/to/weights  # custom weights dir

Run this BEFORE starting the server for the first time.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

# ─── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# ─── Model Registry ─────────────────────────────────────────────────────────────
# Each entry: (name, sub_dir_or_none, download_fn, size_hint, description)
# download_fn receives (weights_dir: Path) and returns True on success / skip.

MODELS = {}


def register(name: str, size_hint: str, description: str):
    """Decorator to register a model downloader."""
    def decorator(fn):
        MODELS[name] = {
            "fn": fn,
            "size": size_hint,
            "desc": description,
        }
        return fn
    return decorator


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress display. Returns True on success."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already exists ({size_mb:.1f} MB): {dest.name}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"  ↓ Downloading {desc or dest.name}...")
    print(f"    URL: {url}")
    print(f"    To:  {dest}")

    start_time = time.time()

    def _progress(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            elapsed = time.time() - start_time
            speed = mb / max(elapsed, 0.01)
            sys.stdout.write(f"\r    {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB  ({speed:.1f} MB/s)  ")
        else:
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r    {mb:.1f} MB downloaded...")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(tmp), reporthook=_progress)
        print()  # newline after progress
        tmp.rename(dest)
        elapsed = time.time() - start_time
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Done ({size_mb:.1f} MB in {elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def _download_hf_repo(repo_id: str, cache_dir: Path, desc: str = "") -> bool:
    """Download a HuggingFace model repo using huggingface_hub snapshot_download."""
    print(f"  ↓ Downloading {desc or repo_id}...")
    print(f"    Repo: {repo_id}")
    print(f"    Cache: {cache_dir}")

    start_time = time.time()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        elapsed = time.time() - start_time
        print(f"  ✓ Done ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


# ─── Model Downloaders ─────────────────────────────────────────────────────────

@register("grounding_dino", "~700 MB", "IDEA-Research/grounding-dino-base (object detection)")
def dl_grounding_dino(weights_dir: Path):
    cache = weights_dir / "grounding_dino"
    if (cache / "config.json").exists():
        print(f"  ✓ Already exists: {cache}")
        return True
    return _download_hf_repo(
        "IDEA-Research/grounding-dino-base",
        cache,
        "Grounding DINO base model",
    )


@register("sam2", "~900 MB", "SAM2 hiera large (segmentation)")
def dl_sam2(weights_dir: Path):
    dest = weights_dir / "sam2" / "sam2_hiera_large.pt"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already exists ({size_mb:.1f} MB): {dest.name}")
        return True
    return _download_file(
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        dest,
        "SAM2 hiera large checkpoint",
    )


@register("sam", "~2.4 GB", "SAM vit_h (fallback segmentation)")
def dl_sam(weights_dir: Path):
    dest = weights_dir / "sam" / "sam_vit_h_4b8939.pth"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already exists ({size_mb:.1f} MB): {dest.name}")
        return True
    return _download_file(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        dest,
        "SAM vit_h checkpoint",
    )


@register("qwen", "~1 GB", "Qwen2.5-0.5B-Instruct (LLM prompt parser)")
def dl_qwen(weights_dir: Path):
    cache = weights_dir / "qwen"
    if (cache / "config.json").exists() or any(cache.glob("*.safetensors")):
        print(f"  ✓ Already exists: {cache}")
        return True
    return _download_hf_repo(
        "Qwen/Qwen2.5-0.5B-Instruct",
        cache,
        "Qwen2.5-0.5B-Instruct LLM",
    )


@register("sdxl_inpaint", "~6 GB", "SDXL 1.0 Inpainting (generative edits)")
def dl_sdxl_inpaint(weights_dir: Path):
    cache = weights_dir / "sdxl_inpaint"
    if (cache / "model_index.json").exists():
        print(f"  ✓ Already exists: {cache}")
        return True
    return _download_hf_repo(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        cache,
        "SDXL inpainting pipeline",
    )


@register("sdxl_img2img", "~6 GB", "SDXL 1.0 Base (style transfer)")
def dl_sdxl_img2img(weights_dir: Path):
    cache = weights_dir / "sdxl_img2img"
    if (cache / "model_index.json").exists():
        print(f"  ✓ Already exists: {cache}")
        return True
    return _download_hf_repo(
        "stabilityai/stable-diffusion-xl-base-1.0",
        cache,
        "SDXL base img2img pipeline",
    )


@register("flux_fill", "~24 GB", "FLUX.1 Fill dev (background reconstruction)")
def dl_flux_fill(weights_dir: Path):
    cache = weights_dir / "flux_fill"
    if (cache / "model_index.json").exists():
        print(f"  ✓ Already exists: {cache}")
        return True
    return _download_hf_repo(
        "black-forest-labs/FLUX.1-fill-dev",
        cache,
        "FLUX.1 Fill dev pipeline",
    )


@register("realesrgan", "~65 MB", "Real-ESRGAN x2 (upscaling)")
def dl_realesrgan(weights_dir: Path):
    dest = weights_dir / "realesrgan" / "RealESRGAN_x2plus.pth"
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already exists ({size_mb:.1f} MB): {dest.name}")
        return True
    return _download_file(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        dest,
        "Real-ESRGAN x2plus weights",
    )


@register("rembg", "~170 MB", "rembg u2net (background removal)")
def dl_rembg(weights_dir: Path):
    """rembg downloads its own ONNX models on first use — trigger that."""
    print("  ↓ Triggering rembg u2net model download...")
    try:
        from rembg import new_session
        session = new_session(
            "u2net",
            providers=["CPUExecutionProvider"],
        )
        print("  ✓ rembg u2net model downloaded and ready")
        return True
    except Exception as e:
        print(f"  ✗ rembg download failed: {e}")
        return False


@register("lama", "~200 MB", "LaMa inpainting (background fallback)")
def dl_lama(weights_dir: Path):
    """LaMa downloads via simple_lama_inpainting on first use."""
    print("  ↓ Triggering LaMa model download...")
    try:
        from simple_lama_inpainting import SimpleLama
        lama = SimpleLama()
        print("  ✓ LaMa model downloaded and ready")
        return True
    except Exception as e:
        print(f"  ✗ LaMa download failed (non-critical, cv2 fallback available): {e}")
        return True  # non-critical


@register("easyocr", "~100 MB", "EasyOCR (text detection)")
def dl_easyocr(weights_dir: Path):
    """EasyOCR downloads its own models on first use."""
    print("  ↓ Triggering EasyOCR model download...")
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("  ✓ EasyOCR English model downloaded and ready")
        return True
    except Exception as e:
        print(f"  ✗ EasyOCR download failed (non-critical): {e}")
        return True  # non-critical


# ─── Main ───────────────────────────────────────────────────────────────────────

# Order matters: download smaller/essential models first, big ones last
DOWNLOAD_ORDER = [
    "grounding_dino",
    "sam2",
    "sam",
    "qwen",
    "realesrgan",
    "rembg",
    "lama",
    "easyocr",
    "sdxl_inpaint",
    "sdxl_img2img",
    "flux_fill",
]


def list_models():
    """Print all available models with sizes."""
    print("\n" + "=" * 60)
    print("  AI Image Editor — Model Registry")
    print("=" * 60)
    for name in DOWNLOAD_ORDER:
        info = MODELS[name]
        print(f"  {name:<16}  {info['size']:<12}  {info['desc']}")
    print("=" * 60)
    total = "~42 GB (all models)"
    print(f"  Total estimated: {total}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download all AI models for the Image Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all models without downloading",
    )
    parser.add_argument(
        "--only", action="append", default=[],
        help="Download only these models (repeatable, e.g. --only sam2 --only qwen)",
    )
    parser.add_argument(
        "--skip", action="append", default=[],
        help="Skip these models (repeatable, e.g. --skip flux_fill)",
    )
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help=f"Custom weights directory (default: {DEFAULT_WEIGHTS_DIR})",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()
    weights_dir = Path(args.weights_dir) if args.weights_dir else DEFAULT_WEIGHTS_DIR

    if args.list:
        list_models()
        return

    # Determine which models to download
    if args.only:
        to_download = [m for m in args.only if m in MODELS]
        invalid = [m for m in args.only if m not in MODELS]
        if invalid:
            print(f"⚠ Unknown models: {invalid}")
            print(f"  Available: {list(MODELS.keys())}")
            sys.exit(1)
    else:
        to_download = [m for m in DOWNLOAD_ORDER if m not in args.skip]

    print()
    print("=" * 60)
    print("  AI Image Editor — Model Downloader")
    print("=" * 60)
    print(f"  Weights dir: {weights_dir}")
    print(f"  Models to download: {len(to_download)}")
    print(f"  Models: {', '.join(to_download)}")
    print("=" * 60)
    print()

    weights_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_start = time.time()

    for i, name in enumerate(to_download, 1):
        info = MODELS[name]
        print(f"\n[{i}/{len(to_download)}] {name} — {info['desc']}")
        print(f"  Size hint: {info['size']}")
        print("-" * 50)

        start = time.time()
        try:
            success = info["fn"](weights_dir)
        except Exception as e:
            print(f"  ✗ UNEXPECTED ERROR: {e}")
            success = False

        elapsed = time.time() - start
        results[name] = {"success": success, "time": elapsed}
        print(f"  {'✓' if success else '✗'} {name} — {elapsed:.1f}s")

    # Summary
    total_elapsed = time.time() - total_start
    succeeded = sum(1 for r in results.values() if r["success"])
    failed = sum(1 for r in results.values() if not r["success"])

    print()
    print("=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        status = "✓ OK" if r["success"] else "✗ FAILED"
        print(f"  {status}  {name:<18}  {r['time']:.1f}s")
    print("-" * 60)
    print(f"  Total: {succeeded}/{len(results)} succeeded, {failed} failed")
    print(f"  Time:  {total_elapsed:.1f}s")
    print(f"  Dir:   {weights_dir}")
    print("=" * 60)

    if failed:
        print(f"\n⚠ {failed} model(s) failed to download.")
        print("  You can re-run this script — existing models will be skipped.")
        sys.exit(1)
    else:
        print("\n✓ All models downloaded successfully!")
        print("  You can now start the server with: ENV=local uvicorn main:app --reload")


if __name__ == "__main__":
    main()
