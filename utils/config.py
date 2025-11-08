"""
Shared configuration constants for data directories.
Organ별 구조:
    ./datas/<organ>/features
    ./datas/<organ>/labels
"""
from __future__ import annotations

import os
from pathlib import Path

# 프로젝트 루트 (이 파일 기준으로 상위)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 기본 데이터 루트: ./datas
_DATA_ROOT = Path(os.environ.get("POT_DATA_ROOT", _PROJECT_ROOT / "datas")).resolve()

# organ별 하위 폴더 이름
_ORGANS = ["lung", "liver"]

# organ별 features / labels 디렉터리 생성
FEATURES_DIRS = {}
LABELS_DIRS = {}

for organ in _ORGANS:
    organ_root = _DATA_ROOT / organ
    feat_dir = Path(os.environ.get(f"POT_FEATURES_DIR_{organ.upper()}", organ_root / "features")).resolve()
    lab_dir = Path(os.environ.get(f"POT_LABELS_DIR_{organ.upper()}", organ_root / "labels")).resolve()
    feat_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    FEATURES_DIRS[organ] = feat_dir
    LABELS_DIRS[organ] = lab_dir

# backward compatibility (기존 코드와 호환)
# 만약 organ 구분 없이 쓰면 lung 기준으로 fallback
FEATURES_DIR = FEATURES_DIRS["lung"]
LABELS_DIR = LABELS_DIRS["lung"]

__all__ = ["FEATURES_DIRS", "LABELS_DIRS", "FEATURES_DIR", "LABELS_DIR"]
