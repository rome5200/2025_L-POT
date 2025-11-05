import os
import tempfile
from pathlib import Path
from typing import Union, Optional
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from utils.config import FEATURES_DIR, LABELS_DIR
from utils.dicom_processor import DICOMProcessor
from utils.mesh_processor import MeshProcessor


class ProcessingThread(QThread):
    """DICOM 업로드 후 ID로 <ID>_features.npy / <ID>_vertex_labels.npy 를 찾아 예측"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder_path: Union[str, Path], model_manager):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.model_manager = model_manager
        self.features_dir = FEATURES_DIR
        self.labels_dir = LABELS_DIR

    def _extract_id_from_folder(self) -> str:
        name = self.folder_path.name
        if "-" in name:
            return name.split("-")[-1]
        return name

    def _get_feature_path(self, pid: str) -> Optional[Path]:
        cand = self.features_dir / f"{pid}_features.npy"
        return cand if cand.exists() else None

    def _get_label_path(self, pid: str) -> Optional[Path]:
        cand = self.labels_dir / f"{pid}_vertex_labels.npy"
        return cand if cand.exists() else None

    def run(self):
        try:
            self.progress.emit(10, "DICOM 파일 로드 중...")

            with tempfile.TemporaryDirectory() as tmpdir:
                dcm_proc = DICOMProcessor()
                dcm_result = dcm_proc.load_and_process(str(self.folder_path))
                image = dcm_result["image"]
                slices = dcm_result["slices"]

                pid = self._extract_id_from_folder()
                feature_path = self._get_feature_path(pid)
                label_path = self._get_label_path(pid)

                if feature_path is not None:
                    self.progress.emit(30, f"사전 계산 특징 발견: {feature_path.name}")
                    features = np.load(str(feature_path)).astype(np.float32, copy=False)

                    if features.ndim != 2 or features.shape[1] < 3:
                        raise ValueError("피처 데이터 형태가 잘못되었습니다: %r" % (features.shape,))

                    verts = features[:, :3].astype(np.float32, copy=False)

                    self.progress.emit(50, "모델 예측 중...")
                    preds = self.model_manager.predict_segmentation(features)

                    labels = None
                    if label_path is not None:
                        lab = np.load(str(label_path))
                        lab = np.asarray(lab).astype(bool)
                        if lab.size == verts.shape[0]:
                            labels = lab
                            self.progress.emit(60, f"라벨 로드 완료: {label_path.name}")
                        else:
                            self.progress.emit(60, f"라벨 길이가 맞지 않아 무시: {lab.size} vs {verts.shape[0]}")

                    prediction_accuracy = None
                    if labels is not None:
                        prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

                    self.progress.emit(100, "완료!")
                    self.finished.emit({
                        "image": image,
                        "verts": verts,
                        "predictions": preds,
                        "probabilities": None,
                        "labels": labels,
                        "mesh_path": None,
                        "model_accuracy": getattr(self.model_manager, "model_accuracy", None),
                        "prediction_accuracy": prediction_accuracy,
                        "selected_folder": self.folder_path.name,
                        "feature_file": str(feature_path),
                    })
                    return

                # feature 못 찾으면 기존 파이프라인
                self.progress.emit(40, "사전 feature가 없어 메쉬를 생성합니다...")
                mesh_proc = MeshProcessor()
                mesh, mesh_path = mesh_proc.create_mesh(image, slices, tmpdir)

                self.progress.emit(60, "정점 특징 추출 중...")
                verts, feat, transform, dists = mesh_proc.extract_features(mesh)

                self.progress.emit(70, "모델 예측 중...")
                preds, probs = self.model_manager.predict_segmentation(
                    feat, return_probabilities=True
                )

                labels = None
                if label_path is not None:
                    lab = np.load(str(label_path)).astype(bool)
                    if lab.size == verts.shape[0]:
                        labels = lab

                prediction_accuracy = None
                if labels is not None and labels.size == preds.size:
                    prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

                self.progress.emit(100, "완료!")
                self.finished.emit({
                    "image": image,
                    "verts": verts,
                    "predictions": preds,
                    "probabilities": probs,
                    "labels": labels,
                    "mesh_path": mesh_path,
                    "model_accuracy": getattr(self.model_manager, "model_accuracy", None),
                    "prediction_accuracy": prediction_accuracy,
                    "selected_folder": self.folder_path.name,
                    "feature_file": None,
                })

        except Exception as e:
            self.error.emit("처리 중 오류 발생: %s" % str(e))


class UploadPage(QWidget):
    """DICOM 폴더를 고르면 위의 파이프라인을 태우는 페이지"""

    # ✅ 여기에 시그널 추가
    processing_completed = pyqtSignal()

    def __init__(self, model_manager, data_store: dict):
        super().__init__()
        self.model_manager = model_manager
        self.data_store = data_store
        self.processing_thread = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        info = QLabel("분석할 DICOM 폴더를 선택하세요")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(12)
        info.setFont(f)
        layout.addWidget(info)

        self.btn_select = QPushButton("DICOM 폴더 선택")
        self.btn_select.setMinimumHeight(50)
        self.btn_select.clicked.connect(self._select_folder)
        layout.addWidget(self.btn_select)

        self.lbl_filename = QLabel("")
        self.lbl_filename.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_filename)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setVisible(False)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM 폴더 선택", "")
        if folder:
            self.lbl_filename.setText(f"선택된 폴더: {os.path.basename(folder)}")
            self._start_processing(folder)

    def _start_processing(self, folder_path: str):
        self.btn_select.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_area.setVisible(True)
        self.log_area.clear()

        self.data_store.clear()
        self.data_store["selected_folder"] = Path(folder_path).name

        self.processing_thread = ProcessingThread(folder_path, self.model_manager)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _on_progress(self, val: int, msg: str):
        self.progress.setValue(val)
        self.log_area.append(f"[{val}%] {msg}")

    def _on_finished(self, result: dict):
        self.data_store.update(result)
        self.log_area.append("\n처리가 완료되었습니다.")
        self.btn_select.setEnabled(True)
        # ✅ 메인윈도우로 알려주기
        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.log_area.append(f"\n오류: {msg}")
        self.btn_select.setEnabled(True)
