from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _collect_videos(ffpp_root: Path, limit_per_class: int, seed: int) -> list[dict[str, Any]]:
    real_paths = sorted((ffpp_root / "original").glob("*.mp4"))
    fake_paths: list[Path] = []
    for fake_dir in ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]:
        fake_paths.extend(sorted((ffpp_root / fake_dir).glob("*.mp4")))

    if not real_paths:
        raise RuntimeError(f"No real videos found under {ffpp_root / 'original'}")
    if not fake_paths:
        raise RuntimeError("No fake videos found under FF++ manipulation directories")

    rng = random.Random(seed)
    real_sample = rng.sample(real_paths, min(limit_per_class, len(real_paths)))
    fake_sample = rng.sample(fake_paths, min(limit_per_class, len(fake_paths)))

    picked: list[dict[str, Any]] = (
        [{"path": p, "label": "REAL"} for p in real_sample]
        + [{"path": p, "label": "FAKE"} for p in fake_sample]
    )
    rng.shuffle(picked)
    return picked


def _post_analyze(base_url: str, video_path: Path, n_frames: int, timeout_s: int) -> tuple[int, dict[str, Any]]:
    with video_path.open("rb") as f:
        resp = requests.post(
            f"{base_url}/analyze",
            files={"video": (video_path.name, f, "video/mp4")},
            data={"n_frames": str(n_frames)},
            timeout=timeout_s,
        )
    payload: dict[str, Any]
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": resp.text[:1000]}
    return resp.status_code, payload


def _build_report(
    *,
    run_meta: dict[str, Any],
    health: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    ok_records = [r for r in records if r.get("status_code") == 200 and isinstance(r.get("response"), dict)]
    verdicts = Counter(r["response"].get("verdict", "UNKNOWN") for r in ok_records)
    probs = [float(r["response"].get("prob_fake_mean", 0.0)) for r in ok_records]

    real_ok = [r for r in ok_records if r["ground_truth"] == "REAL"]
    fake_ok = [r for r in ok_records if r["ground_truth"] == "FAKE"]

    def _safe_mean(vals: list[float]) -> float | None:
        if not vals:
            return None
        return round(sum(vals) / len(vals), 6)

    real_probs = [float(r["response"].get("prob_fake_mean", 0.0)) for r in real_ok]
    fake_probs = [float(r["response"].get("prob_fake_mean", 0.0)) for r in fake_ok]

    false_fake = sum(1 for r in real_ok if r["response"].get("verdict") == "FAKE")
    false_real = sum(1 for r in fake_ok if r["response"].get("verdict") == "REAL")

    collapsed_one_class = len({r["response"].get("verdict") for r in ok_records}) <= 1 if ok_records else False
    all_probs_side = None
    if probs and "threshold" in health and "uncertain_band" in health:
        hi = min(1.0, float(health["threshold"]) + float(health["uncertain_band"]))
        lo = max(0.0, float(health["threshold"]) - float(health["uncertain_band"]))
        if all(p >= hi for p in probs):
            all_probs_side = "all_above_hi"
        elif all(p <= lo for p in probs):
            all_probs_side = "all_below_lo"

    diagnosis: list[str] = []
    if collapsed_one_class:
        diagnosis.append("All analyzed samples map to one verdict class.")
    if health.get("threshold") is not None and float(health["threshold"]) <= 0.1:
        diagnosis.append(
            f"Threshold is very low ({float(health['threshold']):.4f}); with uncertain band this makes FAKE easy to trigger."
        )
    if all_probs_side == "all_above_hi":
        diagnosis.append("All prob_fake_mean values are above verdict_hi, consistent with class collapse to FAKE.")
    if all_probs_side == "all_below_lo":
        diagnosis.append("All prob_fake_mean values are below verdict_lo, consistent with class collapse to REAL.")

    return {
        "run_meta": run_meta,
        "health": health,
        "counts": {
            "total_attempted": len(records),
            "successful": len(ok_records),
            "failed": len(failures),
            "real_successful": len(real_ok),
            "fake_successful": len(fake_ok),
        },
        "verdict_distribution": dict(verdicts),
        "prob_fake_mean_stats": {
            "overall_mean": _safe_mean(probs),
            "real_mean": _safe_mean(real_probs),
            "fake_mean": _safe_mean(fake_probs),
            "overall_min": round(min(probs), 6) if probs else None,
            "overall_max": round(max(probs), 6) if probs else None,
        },
        "error_rates": {
            "false_FAKE_count": false_fake,
            "false_FAKE_rate_over_real": round(false_fake / len(real_ok), 6) if real_ok else None,
            "false_REAL_count": false_real,
            "false_REAL_rate_over_fake": round(false_real / len(fake_ok), 6) if fake_ok else None,
        },
        "collapse_check": {
            "collapsed_to_one_verdict": collapsed_one_class,
            "all_probs_side_of_decision_band": all_probs_side,
            "likely_cause_evidence": diagnosis,
        },
        "failures": failures,
        "samples": records,
    }


def _write_markdown(report: dict[str, Any], output_md: Path) -> None:
    c = report["counts"]
    p = report["prob_fake_mean_stats"]
    e = report["error_rates"]
    collapse = report["collapse_check"]
    health = report["health"]

    lines = [
        "# E2E /analyze Validation Report",
        "",
        f"- UTC timestamp: `{report['run_meta']['timestamp_utc']}`",
        f"- Base URL: `{report['run_meta']['base_url']}`",
        f"- FF++ root: `{report['run_meta']['ffpp_root']}`",
        f"- Requested sample size: `{report['run_meta']['limit_per_class']} real + {report['run_meta']['limit_per_class']} fake`",
        f"- n_frames per request: `{report['run_meta']['n_frames']}`",
        "",
        "## Backend Policy Snapshot",
        "",
        f"- threshold: `{health.get('threshold')}`",
        f"- uncertain_band: `{health.get('uncertain_band')}`",
        f"- verdict_hi: `{health.get('verdict_hi')}`",
        f"- verdict_lo: `{health.get('verdict_lo')}`",
        "",
        "## Outcome Metrics",
        "",
        f"- successful analyses: `{c['successful']}/{c['total_attempted']}`",
        f"- verdict_distribution: `{report['verdict_distribution']}`",
        f"- prob_fake_mean overall mean/min/max: `{p['overall_mean']} / {p['overall_min']} / {p['overall_max']}`",
        f"- prob_fake_mean real_mean vs fake_mean: `{p['real_mean']} vs {p['fake_mean']}`",
        f"- false-FAKE rate (real->FAKE): `{e['false_FAKE_rate_over_real']}` ({e['false_FAKE_count']} cases)",
        f"- false-REAL rate (fake->REAL): `{e['false_REAL_rate_over_fake']}` ({e['false_REAL_count']} cases)",
        "",
        "## Collapse Check",
        "",
        f"- collapsed_to_one_verdict: `{collapse['collapsed_to_one_verdict']}`",
        f"- all_probs_side_of_decision_band: `{collapse['all_probs_side_of_decision_band']}`",
        f"- likely_cause_evidence: `{collapse['likely_cause_evidence']}`",
        "",
        "## Repro Commands",
        "",
        "```bash",
        "python scripts/validate_e2e_threshold_policy.py \\",
        "  --ffpp-root \"data/FaceForensics++_C23\" \\",
        "  --base-url \"http://127.0.0.1:8000\" \\",
        "  --limit-per-class 20 \\",
        "  --n-frames 12 \\",
        "  --seed 42",
        "```",
        "",
        f"JSON artifact: `{report['run_meta']['json_artifact']}`",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate /analyze behavior on mixed FF++ real/fake videos.")
    parser.add_argument("--ffpp-root", default="data/FaceForensics++_C23")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit-per-class", type=int, default=20)
    parser.add_argument("--n-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    ffpp_root = Path(args.ffpp_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_json = out_dir / f"validate_e2e_threshold_policy_{ts}.json"
    output_md = out_dir / f"validate_e2e_threshold_policy_{ts}.md"

    health_resp = requests.get(f"{args.base_url}/health", timeout=10)
    health_resp.raise_for_status()
    health = health_resp.json()

    picked = _collect_videos(ffpp_root, limit_per_class=args.limit_per_class, seed=args.seed)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in picked:
        video_path: Path = item["path"]
        gt = item["label"]
        status_code, payload = _post_analyze(args.base_url, video_path, args.n_frames, args.timeout_s)
        record = {
            "ground_truth": gt,
            "video_path": str(video_path),
            "status_code": status_code,
            "response": payload,
        }
        records.append(record)
        if status_code != 200:
            failures.append(record)

    run_meta = {
        "timestamp_utc": ts,
        "base_url": args.base_url,
        "ffpp_root": str(ffpp_root),
        "limit_per_class": args.limit_per_class,
        "n_frames": args.n_frames,
        "seed": args.seed,
        "json_artifact": str(output_json),
        "md_artifact": str(output_md),
    }

    report = _build_report(run_meta=run_meta, health=health, records=records, failures=failures)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, output_md)

    print(json.dumps({
        "json_artifact": str(output_json),
        "md_artifact": str(output_md),
        "successful": report["counts"]["successful"],
        "attempted": report["counts"]["total_attempted"],
        "verdict_distribution": report["verdict_distribution"],
        "collapse_check": report["collapse_check"],
    }, indent=2))


if __name__ == "__main__":
    main()
