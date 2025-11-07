"""Qt 전용 파이프라인 실행 스레드."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from PyQt6.QtCore import QThread, pyqtSignal


class PipelineRunnerThread(QThread):
    """공통 파이프라인을 백그라운드에서 실행하는 ``QThread``.

    Parameters
    ----------
    pipeline:
        ``pipelines.common_pipeline.BaseCTPipeline`` 또는 호환되는 객체.
    folder_path:
        DICOM 파일이 들어 있는 폴더 경로.
    model_manager:
        ``predict_for_organ`` 메서드를 제공하는 모델 매니저.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        pipeline: Any,
        folder_path: Union[str, Path],
        model_manager: Any,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._folder_path = Path(folder_path)
        self._model_manager = model_manager

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------
    def _on_progress(self, percentage: int, message: str) -> None:
        self.progress.emit(int(percentage), str(message))

    # ------------------------------------------------------------------
    # QThread 구현
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - Qt 스레드 실행부
        try:
            result: Dict[str, Any] = self._pipeline.run(
                str(self._folder_path),
                self._model_manager,
                progress_cb=self._on_progress,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - UI에 그대로 노출
            self.error.emit(str(exc))
