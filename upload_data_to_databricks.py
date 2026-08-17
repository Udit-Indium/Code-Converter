from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUTS_DIR = pathlib.Path(__file__).parent / "outputs"
PATHS_FILE = OUTPUTS_DIR / "databricks_data_paths.json"

# Anything a data pipeline is likely to read. Extend if your sources differ.
DATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson",
                 ".parquet", ".orc", ".avro", ".xlsx", ".xls"}

_REQUEST_TIMEOUT = 300


def _config(volume_override: "str | None") -> tuple[str, dict, str]:
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_API_KEY")
    volume = volume_override or os.getenv("DATABRICKS_VOLUME")

    missing = [
        name
        for name, value in (("DATABRICKS_HOST", host),
                            ("DATABRICKS_API_KEY", token),
                            ("DATABRICKS_VOLUME (or --volume)", volume))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing: {', '.join(missing)}. The volume should look like "
            "/Volumes/<catalog>/<schema>/<volume>."
        )
    if not volume.startswith("/Volumes/"):
        raise RuntimeError(
            f"Volume must start with /Volumes/ — got {volume!r}. "
            "Create it in Unity Catalog first."
        )
    return host.rstrip("/"), {"Authorization": f"Bearer {token}"}, volume.rstrip("/")


def find_data_files(local_dir: pathlib.Path) -> list[pathlib.Path]:
    """Every data file under `local_dir`, recursively, sorted for stable output."""
    return sorted(
        p for p in local_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in DATA_SUFFIXES
    )


def upload(local_dir: pathlib.Path, volume_override: "str | None" = None) -> dict:
    """Upload every data file under `local_dir` to the volume.

    Subdirectories are preserved, so `data/raw/sales.csv` lands at
    `<volume>/raw/sales.csv`. Uses the Files API with overwrite=true, so
    re-running is idempotent.

    Returns:
        {"remote_dir": <volume>, "files": {<relative path>: <remote path>}}
    """
    host, headers, volume = _config(volume_override)

    files = find_data_files(local_dir)
    if not files:
        raise RuntimeError(
            f"No data files found under {local_dir} "
            f"(looked for: {', '.join(sorted(DATA_SUFFIXES))})."
        )

    uploaded: dict[str, str] = {}
    for path in files:
        rel = path.relative_to(local_dir).as_posix()
        remote = f"{volume}/{rel}"
        with path.open("rb") as fh:
            r = requests.put(
                f"{host}/api/2.0/fs/files{remote}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                params={"overwrite": "true"},
                data=fh,
                timeout=_REQUEST_TIMEOUT,
            )
        r.raise_for_status()
        uploaded[rel] = remote
        print(f"{rel:<40} {path.stat().st_size:>10} bytes -> {remote}")

    return {"remote_dir": volume, "files": uploaded}


def write_paths_file(mapping: dict) -> pathlib.Path:
    """Persist the mapping the converter agent reads to rewrite paths."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PATHS_FILE.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return PATHS_FILE


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload a source script's input data to a Databricks volume."
    )
    ap.add_argument("data_dir", help="Local directory holding the input data files.")
    ap.add_argument("--volume", default=None,
                    help="Target volume; overrides DATABRICKS_VOLUME.")
    args = ap.parse_args()

    local_dir = pathlib.Path(args.data_dir).expanduser().resolve()
    if not local_dir.is_dir():
        print(f"Not a directory: {local_dir}", file=sys.stderr)
        return 1

    try:
        mapping = upload(local_dir, args.volume)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    written = write_paths_file(mapping)
    print(f"\n{len(mapping['files'])} file(s) uploaded to {mapping['remote_dir']}")
    print(f"Mapping written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
