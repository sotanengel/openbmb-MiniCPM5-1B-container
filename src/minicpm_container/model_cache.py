"""Prepare and validate local MiniCPM5 model directories for Docker builds."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_CONTAINER_MODEL_PATH = "/models/MiniCPM5-1B"


def is_complete_model_dir(path: Path) -> bool:
    """Return True when the directory contains config and weight artifacts."""
    if not path.is_dir():
        return False

    config_path = path / "config.json"
    if not config_path.is_file():
        return False

    if any(path.glob("*.safetensors")):
        return True

    return (path / "model.safetensors.index.json").is_file()


def _hf_hub_cache_root() -> Path:
    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(cache)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub"


def _repo_cache_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def find_hf_cache_snapshot(repo_id: str) -> Path | None:
    """Return the newest complete HF cache snapshot for repo_id, if any."""
    snapshots_root = _hf_hub_cache_root() / _repo_cache_dir_name(repo_id) / "snapshots"
    if not snapshots_root.is_dir():
        return None

    candidates = [
        snapshot
        for snapshot in snapshots_root.iterdir()
        if snapshot.is_dir() and is_complete_model_dir(snapshot)
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _copy_file_with_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def materialize_model_dir(source: Path, target: Path) -> None:
    """Copy or hardlink model files into target, resolving symlinks."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.iterdir()):
        destination = target / item.name
        if item.is_symlink():
            resolved = item.resolve()
            if resolved.is_dir():
                shutil.copytree(resolved, destination, symlinks=False)
            else:
                _copy_file_with_hardlink(resolved, destination)
        elif item.is_dir():
            shutil.copytree(
                item,
                destination,
                symlinks=False,
                copy_function=_copy_file_with_hardlink,
            )
        else:
            _copy_file_with_hardlink(item, destination)


def _docker_image_exists(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def extract_model_from_image(
    image: str,
    target: Path,
    *,
    container_path: str = DEFAULT_CONTAINER_MODEL_PATH,
) -> bool:
    """Extract model weights from an existing Docker image into target."""
    if not _docker_image_exists(image):
        return False

    target.mkdir(parents=True, exist_ok=True)
    container_id = ""
    try:
        create = subprocess.run(
            ["docker", "create", image],
            check=True,
            capture_output=True,
            text=True,
        )
        container_id = create.stdout.strip()
        if not container_id:
            logger.error("docker create returned empty container id for %s", image)
            return False

        subprocess.run(
            ["docker", "cp", f"{container_id}:{container_path}/.", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to extract model from %s: %s", image, exc.stderr or exc)
        return False
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", container_id],
                check=False,
                capture_output=True,
                text=True,
            )

    return is_complete_model_dir(target)


def ensure_model_dir(
    model_dir: Path,
    repo_id: str = DEFAULT_MODEL_ID,
    *,
    image: str | None = None,
) -> bool:
    """Populate model_dir from local sources when incomplete."""
    if is_complete_model_dir(model_dir):
        print(f"Model directory already complete: {model_dir}")
        return True

    model_dir.mkdir(parents=True, exist_ok=True)

    snapshot = find_hf_cache_snapshot(repo_id)
    if snapshot is not None:
        print(f"Materializing model from Hugging Face cache: {snapshot}")
        materialize_model_dir(snapshot, model_dir)
        if is_complete_model_dir(model_dir):
            print(f"Model ready at {model_dir}")
            return True
        logger.error("HF cache snapshot was incomplete after materialize: %s", snapshot)

    if image and extract_model_from_image(image, model_dir):
        print(f"Extracted model from Docker image {image} to {model_dir}")
        return True

    if image and not _docker_image_exists(image):
        print(f"Docker image not found, skipping extraction: {image}")

    print(f"Model directory incomplete: {model_dir}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local MiniCPM5 model directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure", help="Populate MODEL_DIR when possible")
    ensure_parser.add_argument("model_dir", type=Path)
    ensure_parser.add_argument("--repo-id", default=DEFAULT_MODEL_ID)
    ensure_parser.add_argument("--image", default=None)

    check_parser = subparsers.add_parser("check", help="Return success when model_dir is complete")
    check_parser.add_argument("model_dir", type=Path)

    args = parser.parse_args(argv)

    if args.command == "check":
        return 0 if is_complete_model_dir(args.model_dir) else 1

    ready = ensure_model_dir(args.model_dir, args.repo_id, image=args.image)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
