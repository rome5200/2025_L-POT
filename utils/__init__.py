# utils/__init__.py
'''
from .config import FEATURES_DIR, LABELS_DIR
from .helper import resource_path   # ← 이 줄 추가

__all__ = [
    "FEATURES_DIR",
    "LABELS_DIR",
    "resource_path",
]'''
from .helper import resource_path

__all__ = [
    "resource_path",
]
