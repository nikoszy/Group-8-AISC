"""
tests/test_model_registry.py — Unit tests for ModelRegistry.

Run with:
    python -m pytest tests/test_model_registry.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure repo root is on sys.path so `from src.model_registry import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path):
    """Return a ModelRegistry backed by a temp file (cleaned up after each test)."""
    return ModelRegistry(registry_path=tmp_path / "model_registry.json")


def _make_entry(model_id, f1, comparable=True, model_type="lr"):
    return {
        "model_id":      model_id,
        "model_type":    model_type,
        "artifact_path": f"artifacts/{model_id}.pkl",
        "metrics":       {"f1": f1, "precision": 0.8, "recall": 0.7, "auc": 0.85},
        "notes":         f"test entry {model_id}",
        "comparable":    comparable,
    }


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_returns_model_id(self, tmp_registry):
        mid = tmp_registry.register(_make_entry("model_a", 0.70))
        assert mid == "model_a"

    def test_register_persists_to_json(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        entries = json.loads(tmp_registry._path.read_text())
        assert len(entries) == 1
        assert entries[0]["model_id"] == "model_a"

    def test_register_upsert_replaces_existing(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_a", 0.75))  # higher F1 update
        assert len(tmp_registry) == 1
        entries = tmp_registry.list_all()
        assert entries[0]["metrics"]["f1"] == 0.75

    def test_register_stamps_trained_at(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        entries = json.loads(tmp_registry._path.read_text())
        assert "trained_at" in entries[0]
        assert entries[0]["trained_at"]  # non-empty string

    def test_register_missing_required_key_raises(self, tmp_registry):
        bad_entry = {"model_id": "x", "model_type": "lr", "artifact_path": "x.pkl"}
        # missing "metrics"
        with pytest.raises(KeyError):
            tmp_registry.register(bad_entry)

    def test_register_non_dict_metrics_raises(self, tmp_registry):
        bad_entry = {
            "model_id": "x", "model_type": "lr",
            "artifact_path": "x.pkl", "metrics": 0.75
        }
        with pytest.raises(TypeError):
            tmp_registry.register(bad_entry)

    def test_register_defaults_is_active_false(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        entry = tmp_registry.list_all()[0]
        assert entry["is_active"] is False

    def test_register_multiple_entries(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_b", 0.75))
        assert len(tmp_registry) == 2


# ---------------------------------------------------------------------------
# get_best()
# ---------------------------------------------------------------------------

class TestGetBest:
    def test_get_best_returns_highest_f1(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_b", 0.80))
        best = tmp_registry.get_best(metric="f1")
        assert best["model_id"] == "model_b"

    def test_get_best_empty_registry_returns_none(self, tmp_registry):
        assert tmp_registry.get_best() is None

    def test_get_best_excludes_non_comparable(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70, comparable=True))
        tmp_registry.register(_make_entry("model_b", 0.95, comparable=False))
        best = tmp_registry.get_best(metric="f1", only_comparable=True)
        assert best["model_id"] == "model_a"

    def test_get_best_includes_non_comparable_when_flag_false(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70, comparable=True))
        tmp_registry.register(_make_entry("model_b", 0.95, comparable=False))
        best = tmp_registry.get_best(metric="f1", only_comparable=False)
        assert best["model_id"] == "model_b"

    def test_get_best_none_metric_excluded(self, tmp_registry):
        entry = _make_entry("model_a", 0.70)
        entry["metrics"]["f1"] = None
        tmp_registry.register(entry)
        assert tmp_registry.get_best(metric="f1") is None

    def test_get_best_by_auc(self, tmp_registry):
        a = _make_entry("model_a", 0.70)
        a["metrics"]["auc"] = 0.91
        b = _make_entry("model_b", 0.80)
        b["metrics"]["auc"] = 0.88
        tmp_registry.register(a)
        tmp_registry.register(b)
        best_auc = tmp_registry.get_best(metric="auc")
        assert best_auc["model_id"] == "model_a"


# ---------------------------------------------------------------------------
# get_active() / set_active()
# ---------------------------------------------------------------------------

class TestActiveModel:
    def test_get_active_empty_returns_none(self, tmp_registry):
        assert tmp_registry.get_active() is None

    def test_set_active_marks_one_true(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_b", 0.80))
        tmp_registry.set_active("model_b")
        active = tmp_registry.get_active()
        assert active["model_id"] == "model_b"
        assert active["is_active"] is True

    def test_set_active_clears_previous_active(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_b", 0.80))
        tmp_registry.set_active("model_a")
        tmp_registry.set_active("model_b")  # switch
        entries = tmp_registry.list_all()
        active_count = sum(1 for e in entries if e["is_active"])
        assert active_count == 1
        assert tmp_registry.get_active()["model_id"] == "model_b"

    def test_set_active_unknown_id_raises(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        with pytest.raises(KeyError, match="model_z"):
            tmp_registry.set_active("model_z")


# ---------------------------------------------------------------------------
# select_best()
# ---------------------------------------------------------------------------

class TestSelectBest:
    def test_select_best_picks_winner_and_activates(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.register(_make_entry("model_b", 0.85))
        winner = tmp_registry.select_best(metric="f1")
        assert winner["model_id"] == "model_b"
        assert tmp_registry.get_active()["model_id"] == "model_b"

    def test_select_best_empty_returns_none(self, tmp_registry):
        result = tmp_registry.select_best()
        assert result is None

    def test_select_best_excludes_non_comparable(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70, comparable=True))
        tmp_registry.register(_make_entry("model_b", 0.99, comparable=False))
        winner = tmp_registry.select_best(metric="f1")
        assert winner["model_id"] == "model_a"

    def test_select_best_updated_after_new_registration(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        tmp_registry.select_best()
        assert tmp_registry.get_active()["model_id"] == "model_a"

        # Now a better model appears
        tmp_registry.register(_make_entry("model_b", 0.88))
        tmp_registry.select_best()
        assert tmp_registry.get_active()["model_id"] == "model_b"


# ---------------------------------------------------------------------------
# list_all()
# ---------------------------------------------------------------------------

class TestListAll:
    def test_list_all_sorted_descending(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.60))
        tmp_registry.register(_make_entry("model_b", 0.80))
        tmp_registry.register(_make_entry("model_c", 0.70))
        all_entries = tmp_registry.list_all(metric="f1")
        f1_values = [e["metrics"]["f1"] for e in all_entries]
        assert f1_values == sorted(f1_values, reverse=True)

    def test_list_all_none_metric_at_end(self, tmp_registry):
        tmp_registry.register(_make_entry("model_a", 0.70))
        entry_b = _make_entry("model_b", 0.80)
        entry_b["metrics"]["f1"] = None
        tmp_registry.register(entry_b)
        all_entries = tmp_registry.list_all(metric="f1")
        assert all_entries[-1]["model_id"] == "model_b"

    def test_list_all_empty_registry(self, tmp_registry):
        assert tmp_registry.list_all() == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_repr(self, tmp_registry):
        r = repr(tmp_registry)
        assert "ModelRegistry" in r

    def test_len(self, tmp_registry):
        assert len(tmp_registry) == 0
        tmp_registry.register(_make_entry("model_a", 0.70))
        assert len(tmp_registry) == 1

    def test_registry_file_created_on_first_register(self, tmp_registry):
        assert not tmp_registry._path.exists()
        tmp_registry.register(_make_entry("model_a", 0.70))
        assert tmp_registry._path.exists()

    def test_different_model_types(self, tmp_registry):
        tmp_registry.register(_make_entry("lr_001", 0.70, model_type="lr"))
        tmp_registry.register(_make_entry("cnn_001", 0.85, model_type="cnn"))
        tmp_registry.register(_make_entry("stacked_001", 0.88, model_type="stacked"))
        assert len(tmp_registry) == 3
        best = tmp_registry.get_best()
        assert best["model_type"] == "stacked"
