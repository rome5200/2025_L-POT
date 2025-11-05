"""Reusable 3D viewer widget for lung visualization."""
from __future__ import annotations

import logging
import numpy as np
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager


def _configure_korean_font() -> str:
    """한글 폰트를 탐색하고 설정합니다.
    
    Returns:
        str: 설정된 폰트 이름 또는 대체 메시지
    """
    # 한글 폰트 후보 목록 (우선순위 순)
    korean_fonts = [
        'Malgun Gothic',      # 맑은 고딕 (Windows)
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'Noto Sans CJK KR',   # Noto Sans CJK Korean
        'AppleGothic',        # Apple Gothic (macOS)
        'DejaVu Sans',        # 대체 폰트
    ]
    
    # 시스템에 설치된 폰트 목록 가져오기
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 한글 폰트 후보 중에서 사용 가능한 폰트 찾기
    for font_name in korean_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            logging.info(f"한글 폰트 설정 완료: {font_name}")
            return f"한글 폰트 설정 완료: {font_name}"
    
    # 한글 폰트를 찾지 못한 경우 기본 설정
    plt.rcParams["axes.unicode_minus"] = False
    warning_msg = "한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다."
    logging.warning(warning_msg)
    return warning_msg


class Lung3DViewer(QWidget):
    """Qt widget that renders 3D lung point clouds with Matplotlib."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        info_text: str | None = "데이터를 업로드하면 3D 시각화가 표시됩니다.",
        figure_size: tuple[float, float] = (8, 8),
        info_style: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._verts: np.ndarray | None = None
        self._predictions: np.ndarray | None = None
        self._threshold: float = 0.5

        # 한글 폰트 설정 (Figure 생성 전에 호출)
        font_status = _configure_korean_font()
        logging.info(f"Lung3DViewer 초기화 - {font_status}")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = plt.figure(figsize=figure_size)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.info_label: QLabel | None = None
        if info_text:
            self.info_label = QLabel(info_text)
            self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.info_label.setStyleSheet(
                info_style
                or """
                background-color: #eaf6fe;
                border-radius: 10px;
                padding: 20px;
                font-size: 14px;
                """
            )
            layout.addWidget(self.info_label)

        self.setLayout(layout)

    def update_plot(
        self,
        verts: np.ndarray,
        predictions: np.ndarray | None = None,
        ground_truth_mask: np.ndarray | None = None,
        *,
        downsample_step: int | None = None,
        threshold: float = 0.5,
        base_kwargs: dict | None = None,
        highlight_kwargs: dict | None = None,
        gt_kwargs: dict | None = None,
        show_legend: bool = True,
        title: str | None = None,
        axis_labels: tuple[str, str, str] = ("X", "Y", "Z"),
    ) -> None:
        """Render the 3D scatter plot with optional predictions overlay."""
        if verts.size == 0:
            self.clear()
            return

        self._verts = verts
        self._predictions = predictions
        self._threshold = threshold

        if self.info_label:
            self.info_label.setVisible(False)

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")

        sampled_verts = self._apply_downsample(verts, downsample_step)

        base_defaults = {
            "c": "gray",
            "s": 2,
            "alpha": 0.3,
            "label": "폐 전체",
        }
        if base_kwargs:
            base_defaults.update(base_kwargs)

        if sampled_verts.size:
            ax.scatter(
                sampled_verts[:, 0],
                sampled_verts[:, 1],
                sampled_verts[:, 2],
                **base_defaults,
            )

        highlight_defaults = {
            "c": "red",
            "s": 10,
            "alpha": 0.9,
            "label": "예측 결절",
        }
        if highlight_kwargs:
            highlight_defaults.update(highlight_kwargs)

        if predictions is not None:
            prediction_mask = predictions > threshold
            nodule_verts = verts[prediction_mask]
            if nodule_verts.size:
                ax.scatter(
                    nodule_verts[:, 0],
                    nodule_verts[:, 1],
                    nodule_verts[:, 2],
                    **highlight_defaults,
                )

        # Ground Truth 오버레이
        gt_defaults = {
            "c": "blue",
            "s": 4,
            "alpha": 0.9,
            "label": "실제 결절",
        }
        if gt_kwargs:
            gt_defaults.update(gt_kwargs)

        if ground_truth_mask is not None:
            gt_verts = verts[ground_truth_mask]
            if gt_verts.size:
                ax.scatter(
                    gt_verts[:, 0],
                    gt_verts[:, 1],
                    gt_verts[:, 2],
                    **gt_defaults,
                )

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        if show_legend and ax.get_legend_handles_labels()[0]:
            ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self) -> None:
        """Clear the figure and show the info label again if available."""
        self.figure.clear()
        self.canvas.draw()
        if self.info_label:
            self.info_label.setVisible(True)

    @staticmethod
    def _apply_downsample(verts: np.ndarray, downsample_step: int | None) -> np.ndarray:
        if downsample_step is None or downsample_step <= 1:
            return verts

        indices = np.arange(0, len(verts), downsample_step)
        return verts[indices]
