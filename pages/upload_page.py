# pages/upload_page.py

import os
from pathlib import Path
from typing import Union

import numpy as np  # data_store에 들어오는 dict에 numpy가 있을 수 있어서 유지

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from pipelines.lung_pipeline import LungPipeline  # ✅ 우리가 만든 파이프라인
# model_manager는 main에서 만들어서 이 페이지로 넘겨준다고 가정


class ProcessingThread(QThread):
    """
    선택한 DICOM 폴더를 파이프라인에 태워서 결과를 가져오는 쓰레드.
    실제 처리 로직은 pipelines/lung_pipeline.py 안에 있고
    여기서는 그냥 호출 + 시그널만 담당한다.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder_path: Union[str, Path], model_manager):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.model_manager = model_manager
        self.pipeline = LungPipeline()

    def _progress_cb(self, pct: int, msg: str):
        """파이프라인에서 호출할 콜백을 PyQt 시그널로 바꿔주는 함수"""
        self.progress.emit(pct, msg)

    def run(self):
        try:
            # 파이프라인 실행
            result = self.pipeline.run(
                str(self.folder_path),
                self.model_manager,
                progress_cb=self._progress_cb,
            )
            # 성공 시 결과 emit
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"처리 중 오류 발생: {e}")


class UploadPage(QWidget):
    """DICOM 폴더를 고르면 위의 파이프라인을 태우는 페이지"""

    # 메인 윈도우 등에 "처리 끝났다" 알려줄 때 쓰는 신호
    processing_completed = pyqtSignal()

    def __init__(self, model_manager, data_store: dict):
        super().__init__()
        self.model_manager = model_manager   # ✅ main에서 주입
        self.data_store = data_store         # ✅ 처리 결과를 공유하는 dict
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
        # UI 잠그기
        self.btn_select.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.log_area.setVisible(True)
        self.log_area.clear()

        # 결과 저장 dict 초기화
        self.data_store.clear()
        self.data_store["selected_folder"] = Path(folder_path).name

        # 백그라운드 처리 시작
        self.processing_thread = ProcessingThread(folder_path, self.model_manager)
        self.processing_thread.progress.connect(self._on_progress)
        self.processing_thread.finished.connect(self._on_finished)
        self.processing_thread.error.connect(self._on_error)
        self.processing_thread.start()

    def _on_progress(self, val: int, msg: str):
        self.progress.setValue(val)
        self.log_area.append(f"[{val}%] {msg}")

    def _on_finished(self, result: dict):
        # 파이프라인에서 온 dict를 data_store에 저장
        self.data_store.update(result)
        self.log_area.append("\n처리가 완료되었습니다.")
        self.btn_select.setEnabled(True)
        # 메인윈도우 등 다른 곳에 알려주기
        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.log_area.append(f"\n오류: {msg}")
        self.btn_select.setEnabled(True)
