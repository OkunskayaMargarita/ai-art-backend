"""Explicit local model installation; no downloads occur during inference."""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BACKEND = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def install(item, models):
    target = models / item["folder"] / item["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == item["size"] and digest(target) == item["sha256"]:
            print(f"Already installed: {target.name}", flush=True)
            return
        raise RuntimeError(f"Existing file differs; refusing to overwrite: {target}")
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == item["size"]:
            break
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}?download=true&request={int(time.time())}"
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                done = offset if resume else 0
                reported = done
                with partial.open("ab" if resume else "wb") as file:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        file.write(chunk)
                        done += len(chunk)
                        if done - reported >= 256 * 1024 * 1024:
                            print(f"{target.name}: {done / item['size']:.0%}", flush=True)
                            reported = done
            break
        except requests.RequestException:
            if attempt == 2:
                raise
    if partial.stat().st_size != item["size"] or digest(partial) != item["sha256"]:
        raise RuntimeError(f"Checksum mismatch: {partial}")
    os.replace(partial, target)
    print(f"Installed and verified: {target.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path,
                        default=BACKEND.parent / "ComfyUI" / "models")
    args = parser.parse_args()
    manifest = json.loads((BACKEND / "presets/reference_editor_models.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda item: install(item, args.models_dir.resolve()), manifest))
