# pages/ui_viewers.py
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CT2DViewer(QWidget):
    """
    3D CT 볼륨 (z, h, w)을 받아서 슬라이더로 슬라이스를 바꿔가며 보여주는 위젯.
    - set_image(volume) 으로 데이터 주입
    - HU 범위는 기본 (-1000, 400) 으로 윈도잉해서 8bit로 보여줌
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image_data: np.ndarray | None = None
        self._build_ui()

    # -----------------------------------------------------------
    # UI 구성
    # -----------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # 메인 이미지
        self.image_label = QLabel("이미지가 없습니다.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(512, 512)
        self.image_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #2dc9c8;
            border-radius: 5px;
        """)
        root.addWidget(self.image_label, stretch=1)

        # 슬라이더 + 정보
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self.lbl_info = QLabel("슬라이스: - / -")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 12px;")
        bottom.addWidget(self.lbl_info)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #e9fcff;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2dc9c8;
                border: 1px solid #2dc9c8;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        bottom.addWidget(self.slice_slider, stretch=1)

        root.addLayout(bottom)

        self.setLayout(root)

    # -----------------------------------------------------------
    # 외부에서 호출하는 메서드
    # -----------------------------------------------------------
    def set_image(self, volume: np.ndarray):
        """
        CT 볼륨을 설정한다.
        volume: (z, h, w) or (h, w) ndarray
        """
        if volume is None:
            self._image_data = None
            self.image_label.setText("이미지가 없습니다.")
            self.slice_slider.setEnabled(False)
            self.lbl_info.setText("슬라이스: - / -")
            return

        arr = np.asarray(volume)
        if arr.ndim == 2:
            # (h, w) 만 들어오면 z 차원을 1로 처리
            arr = arr[np.newaxis, ...]

        if arr.ndim != 3:
            raise ValueError("CT2DViewer.set_image() 는 (z, h, w) 형태의 배열만 받을 수 있습니다.")

        self._image_data = arr

        # 슬라이더 설정
        z = arr.shape[0]
        self.slice_slider.setMaximum(z - 1)
        self.slice_slider.setValue(z // 2)
        self.slice_slider.setEnabled(True)

        # 현재 슬라이스 표시
        self._show_slice(self.slice_slider.value())

    # -----------------------------------------------------------
    # 내부 동작
    # -----------------------------------------------------------
    def _on_slice_changed(self, idx: int):
        self._show_slice(idx)

    def _show_slice(self, idx: int):
        if self._image_data is None:
            return

        z, h, w = self._image_data.shape
        idx = max(0, min(idx, z - 1))

        self.lbl_info.setText(f"슬라이스: {idx} / {z - 1}")

        slice_img = self._image_data[idx]

        # HU 윈도잉 (-1000 ~ 400) → 0~255
        slice_img = np.clip(slice_img, -1000, 400)
        norm = (slice_img + 1000) / 1400.0  # 0~1
        norm = (norm * 255).astype(np.uint8)

        # QImage로 변환
        height, width = norm.shape
        bytes_per_line = width
        q_img = QImage(
            norm.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        )

        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            512,
            512,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


class Lung3DViewer(QWidget):
    """
    간단한 3D 폐/결절 시각화용 위젯.
    verts (N, 3)를 그리고, 예측/정답 마스크가 있으면 색을 다르게 표시한다.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_plot(
        self,
        verts: np.ndarray | None,
        predictions: np.ndarray | None = None,
        ground_truth_mask: np.ndarray | None = None,
        title: str = "3D Viewer",
    ):
        """
        verts: (N, 3)
        predictions: bool or 0/1 mask
        ground_truth_mask: bool or 0/1 mask
        """
        self.ax.clear()
        self.ax.set_title(title)

        if verts is None or len(verts) == 0:
            self.ax.text(0.5, 0.5, 0.5, "데이터 없음", color="gray")
            self.canvas.draw()
            return

        # 전체 구조 (희미하게)
        self.ax.scatter(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            c="gray",
            s=1,
            alpha=0.2,
            label="Structure",
        )

        # 예측 결과
        if predictions is not None and np.any(predictions):
            pred_mask = predictions.astype(bool)
            pos = verts[pred_mask]
            self.ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                c="red",
                s=6,
                alpha=0.6,
                label="Prediction",
            )

        # 실제 GT
        if ground_truth_mask is not None and np.any(ground_truth_mask):
            gt_mask = ground_truth_mask.astype(bool)
            gt = verts[gt_mask]
            self.ax.scatter(
                gt[:, 0],
                gt[:, 1],
                gt[:, 2],
                c="cyan",
                s=6,
                alpha=0.6,
                label="Ground Truth",
            )

        self.ax.legend(loc="upper right")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw()
