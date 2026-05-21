"""Tests for derive_is_fake() path-label logic in run_inference.py."""

from pathlib import Path

import pytest

from run_inference import derive_is_fake


@pytest.mark.parametrize(
    "path, expected",
    [
        ("data/manipulated/vid_001", 1),
        ("data/manipulated_sequences/003_004", 1),
        ("/abs/path/to/manipulated/frames/vid", 1),
        ("data/original/vid_002", 0),
        ("data/original_sequences/000", 0),
        ("/abs/path/original/frames/vid", 0),
        ("data/other/vid_003", -1),
        ("data/deepfakes/vid_005", -1),
        ("some/random/path", -1),
        # Both words present — manipulated takes priority
        ("data/original/manipulated/vid_006", 1),
        ("data/manipulated_original_mix/vid_007", 1),
        # Case insensitive
        ("data/Manipulated/vid_008", 1),
        ("data/MANIPULATED/vid_009", 1),
        ("data/Original/vid_010", 0),
        ("data/ORIGINAL/vid_011", 0),
        ("data/MaNiPuLaTeD/vid_012", 1),
    ],
    ids=[
        "manipulated_subdir",
        "manipulated_prefix",
        "manipulated_abs",
        "original_subdir",
        "original_prefix",
        "original_abs",
        "neither_other",
        "neither_deepfakes",
        "neither_random",
        "both_manipulated_wins",
        "both_in_dirname",
        "mixed_case_Manipulated",
        "upper_MANIPULATED",
        "mixed_case_Original",
        "upper_ORIGINAL",
        "sponge_case",
    ],
)
def test_derive_is_fake(path, expected):
    assert derive_is_fake(Path(path)) == expected
