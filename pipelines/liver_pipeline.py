from __future__ import annotations

from pathlib import Path

from pipelines.common_pipeline import OrganCTPipeline
from utils.config import FEATURES_DIR, LABELS_DIR

__all__ = ["LiverPipeline"]


class LiverPipeline(OrganCTPipeline):
    """간 CT 추론을 위한 파이프라인."""

    organ = "liver"

    def __init__(self) -> None:
        super().__init__(self.organ)

        # 간 전용 feature/label 디렉터리를 기본 경로 하위에 구성한다.
        liver_features_dir = (FEATURES_DIR / self.organ).resolve()
        liver_labels_dir = (LABELS_DIR / self.organ).resolve()
        liver_features_dir.mkdir(parents=True, exist_ok=True)
        liver_labels_dir.mkdir(parents=True, exist_ok=True)

        # OrganCTPipeline(BaseCTPipeline)에서 사용하는 경로를 간 전용으로 덮어쓴다.
        self.features_dir: Path = liver_features_dir
        self.labels_dir: Path = liver_labels_dir
