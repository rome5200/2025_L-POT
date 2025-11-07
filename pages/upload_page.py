import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from pipelines import OrganCTPipeline, PipelineRunnerThread

class UploadPage(QWidget):
    """DICOM 폴더를 고르면 위의 파이프라인을 태우는 페이지"""

    # ✅ 여기에 시그널 추가
    processing_completed = pyqtSignal()

    def __init__(self, model_manager, data_store: dict):
        super().__init__()
        self.model_manager = model_manager
        self.data_store = data_store
        self.processing_thread = None
        self.pipeline = OrganCTPipeline("lung")
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

        self.processing_thread = PipelineRunnerThread(
            self.pipeline,
            folder_path,
            self.model_manager,
        )
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
        self.processing_thread = None
        # ✅ 메인윈도우로 알려주기
        self.processing_completed.emit()

    def _on_error(self, msg: str):
        self.log_area.append(f"\n오류: {msg}")
        self.btn_select.setEnabled(True)
        self.processing_thread = None
