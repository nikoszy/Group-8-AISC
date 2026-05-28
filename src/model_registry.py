"""
src/model_registry.py — Central model registry for the deepfake detector.

Reads and writes a JSON manifest at artifacts/model_registry.json.
Every training script (ensemble.py, stacking_ensemble.py, etc.) calls
registry.register() with its eval metrics, then registry.select_best()
to mark the highest-F1 comparable model as active.

The backend (backend/main.py) calls registry.get_active() at startup to
load whichever model currently has the best F1.

Entry schema
------------
{
    "model_id":      str  — unique ID, e.g. "logisticregression_20260527_143022"
    "model_type":    str  — "lr" | "cnn" | "stacked" | "stacked_with_blink"
    "artifact_path": str  — path to .pkl / .pt (relative to repo root)
    "metrics": {
        "f1":        float | None
        "precision": float | None
        "recall":    float | None
        "auc":       float | None
    }
    "trained_at":    str  — ISO-8601 UTC timestamp (set automatically)
    "notes":         str  — free-text description
    "is_active":     bool — exactly one entry is True at a time
    "comparable":    bool — True only if evaluated on the same held-out val
                           split (seed=42, 20%); False entries are excluded
                           from get_best()
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_REGISTRY_PATH = os.path.join("artifacts", "model_registry.json")

_REQUIRED_KEYS = {"model_id", "model_type", "artifact_path", "metrics"}


class ModelRegistry:
    """
    JSON-backed model registry.

    Thread-safety: reads the file fresh on every call, so concurrent
    writer processes will clobber each other.  Fine for our offline
    training workflow (one training script at a time).
    """

    def __init__(self, registry_path: str | os.PathLike = _DEFAULT_REGISTRY_PATH):
        self._path = Path(registry_path)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Return the list of entries from the JSON file, or [] if absent."""
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"Registry file {self._path} must contain a JSON array; got {type(data)}"
            )
        return data

    def _save(self, entries: list[dict]) -> None:
        """Persist the list of entries to the JSON file (human-readable)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, entry: dict[str, Any]) -> str:
        """
        Add or update a model entry in the registry.

        Required keys: model_id, model_type, artifact_path, metrics.
        Optional keys: notes, comparable (default True), is_active (default False).

        Automatically stamps trained_at with the current UTC time if absent.
        If a model with the same model_id already exists, it is replaced
        (upsert semantics).

        Returns the model_id.

        Raises:
            KeyError  — if a required key is missing
            TypeError — if metrics is not a dict
        """
        missing = _REQUIRED_KEYS - set(entry)
        if missing:
            raise KeyError(f"register() missing required keys: {sorted(missing)}")
        if not isinstance(entry.get("metrics"), dict):
            raise TypeError(
                f"entry['metrics'] must be a dict; got {type(entry.get('metrics'))}"
            )

        # Apply defaults
        full_entry: dict[str, Any] = {
            "model_id":      str(entry["model_id"]),
            "model_type":    str(entry["model_type"]),
            "artifact_path": str(entry["artifact_path"]),
            "metrics":       entry["metrics"],
            "trained_at":    entry.get("trained_at") or datetime.now(timezone.utc).isoformat(),
            "notes":         str(entry.get("notes", "")),
            "is_active":     bool(entry.get("is_active", False)),
            "comparable":    bool(entry.get("comparable", True)),
        }

        entries = self._load()
        # Upsert: remove any existing entry with the same model_id
        entries = [e for e in entries if e.get("model_id") != full_entry["model_id"]]
        entries.append(full_entry)
        self._save(entries)
        return full_entry["model_id"]

    def get_best(self, metric: str = "f1", only_comparable: bool = True) -> dict | None:
        """
        Return the entry with the highest value of metrics[metric].

        Args:
            metric          : key inside entry["metrics"] to rank by (default "f1")
            only_comparable : if True (default), only consider entries where
                              comparable=True and metrics[metric] is not None

        Returns the entry dict, or None if the registry is empty / no
        eligible entry exists.
        """
        entries = self._load()

        def _eligible(e: dict) -> bool:
            if only_comparable and not e.get("comparable", True):
                return False
            val = e.get("metrics", {}).get(metric)
            return val is not None

        candidates = [e for e in entries if _eligible(e)]
        if not candidates:
            return None

        return max(candidates, key=lambda e: float(e["metrics"][metric]))

    def get_active(self) -> dict | None:
        """Return the single entry where is_active=True, or None."""
        entries = self._load()
        active = [e for e in entries if e.get("is_active", False)]
        if not active:
            return None
        if len(active) > 1:
            # Defensive: multiple active entries — return the last one written
            # (and let set_active fix the inconsistency next time it's called)
            return active[-1]
        return active[0]

    def set_active(self, model_id: str) -> None:
        """
        Mark model_id as active; set all others to inactive.

        Raises:
            KeyError — if model_id is not found in the registry
        """
        entries = self._load()
        ids = {e.get("model_id") for e in entries}
        if model_id not in ids:
            raise KeyError(
                f"model_id '{model_id}' not found in registry. "
                f"Known IDs: {sorted(ids)}"
            )
        for e in entries:
            e["is_active"] = (e.get("model_id") == model_id)
        self._save(entries)

    def select_best(self, metric: str = "f1") -> dict | None:
        """
        Find the best comparable model by metric, mark it active, return it.

        Convenience wrapper over get_best() + set_active().
        Returns None if the registry has no comparable entries.
        """
        best = self.get_best(metric=metric, only_comparable=True)
        if best is None:
            return None
        self.set_active(best["model_id"])
        return best

    def list_all(self, metric: str = "f1") -> list[dict]:
        """
        Return all entries sorted descending by metrics[metric].

        Entries where metrics[metric] is None are placed at the end.
        """
        entries = self._load()

        def _sort_key(e: dict) -> float:
            val = e.get("metrics", {}).get(metric)
            return float(val) if val is not None else -1.0

        return sorted(entries, key=_sort_key, reverse=True)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._load())

    def __repr__(self) -> str:
        return f"ModelRegistry(path={self._path!r}, n_entries={len(self)})"
