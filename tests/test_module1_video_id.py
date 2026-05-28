"""Tests for module1 video_id → manifest mapping."""

from src.mrl.score import manifest_video_id


def test_real_integer_maps_to_real_prefix():
    assert manifest_video_id("0", is_fake=0) == "real_000"
    assert manifest_video_id("42", is_fake=0) == "real_042"


def test_fake_pair_maps_to_fake_prefix():
    assert manifest_video_id("000_003", is_fake=1) == "fake_000_003"


def test_already_manifest_format_passthrough():
    assert manifest_video_id("real_000", is_fake=0) == "real_000"
    assert manifest_video_id("fake_000_003", is_fake=1) == "fake_000_003"
