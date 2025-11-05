"""Shared configuration constants for data directories."""
from __future__ import annotations

import os
from pathlib import Path

# Base directory for project utilities (i.e., the ``project`` package root)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow overriding the data directories via environment variables for flexibility.
_DATA_ROOT = Path(os.environ.get("POT_DATA_ROOT", _PROJECT_ROOT / "data")).resolve()
FEATURES_DIR = Path(os.environ.get("POT_FEATURES_DIR", _DATA_ROOT / "features")).resolve()
LABELS_DIR = Path(os.environ.get("POT_LABELS_DIR", _DATA_ROOT / "labels")).resolve()

# Ensure the directories exist so downstream code can assume availability.
for _directory in (FEATURES_DIR, LABELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

__all__ = ["FEATURES_DIR", "LABELS_DIR"]
