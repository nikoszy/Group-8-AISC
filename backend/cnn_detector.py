"""
Optional CNN fallback inference extraction.

This file intentionally keeps a light contract-only hook so the backend can
expose fallback activation while ML/CNN runs are still in progress.
"""

from __future__ import annotations

from typing import Any, Callable


def load_cnn_infer() -> Callable[[str], dict[str, Any]] | None:
    """
    Return a callable CNN inference function when available.

    Current Workstream B deliverable keeps this as a non-blocking placeholder.
    Downstream code treats `None` as degraded fallback mode.
    """
    return None
