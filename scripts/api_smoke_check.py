#!/usr/bin/env python3
"""
Small API smoke test for backend /health and /analyze.

Usage examples:
  python scripts/api_smoke_check.py
  python scripts/api_smoke_check.py --base-url http://127.0.0.1:8000 --video path/to/sample.mp4
  python scripts/api_smoke_check.py --output data/experiments/api_smoke_results.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import pathlib
import urllib.error
import urllib.request
import uuid


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _http_json(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url=url, method=method, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body else {}
            return {
                "ok": True,
                "status_code": resp.getcode(),
                "payload": payload,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body) if body else {"detail": body}
        except Exception:
            payload = {"detail": body}
        return {
            "ok": False,
            "status_code": exc.code,
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "payload": {"error": str(exc)},
        }


def _build_multipart(video_path: pathlib.Path, n_frames: int) -> tuple[bytes, str]:
    boundary = f"----CursorApiSmoke{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
    video_bytes = video_path.read_bytes()

    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="n_frames"\r\n\r\n'
        f"{n_frames}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{video_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + video_bytes + tail, boundary


def run_smoke(base_url: str, video: pathlib.Path | None, n_frames: int) -> dict:
    result: dict = {
        "timestamp_utc": _utc_now(),
        "base_url": base_url.rstrip("/"),
        "checks": {},
    }

    health = _http_json(f"{result['base_url']}/health")
    result["checks"]["health"] = health

    if video is None:
        result["checks"]["analyze"] = {
            "ok": False,
            "status_code": None,
            "payload": {"note": "Skipped: no --video path provided."},
        }
        return result

    if not video.exists():
        result["checks"]["analyze"] = {
            "ok": False,
            "status_code": None,
            "payload": {"error": f"Video file does not exist: {video}"},
        }
        return result

    body, boundary = _build_multipart(video, n_frames=n_frames)
    analyze = _http_json(
        f"{result['base_url']}/analyze",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    result["checks"]["analyze"] = analyze
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test /health and /analyze endpoints")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend API base URL")
    parser.add_argument("--video", default="", help="Optional video path for /analyze")
    parser.add_argument("--n-frames", type=int, default=12, help="n_frames form value for /analyze")
    parser.add_argument(
        "--output",
        default="data/experiments/api_smoke_results.json",
        help="Path to write smoke results JSON",
    )
    args = parser.parse_args()

    video_path = pathlib.Path(args.video).resolve() if args.video else None
    results = run_smoke(base_url=args.base_url, video=video_path, n_frames=args.n_frames)

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
