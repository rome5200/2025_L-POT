from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QComboBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

from utils.lung_3d_viewer import Lung3DViewer
from utils.config import FEATURES_DIR, LABELS_DIR


class FeatureLabelViewerPage(QWidget):
    """피처와 라벨 데이터를 비교 시각화하는 페이지"""
    
    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        
        # 데이터 경로 설정
        self.features_dir = FEATURES_DIR
        self.labels_dir = LABELS_DIR
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 제목
        title = QLabel("📌 피처/라벨 비교 뷰어")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 설명
        description = QLabel(
            "사전 처리된 피처와 라벨 데이터를 로드하여 예측 결과와 실제 라벨을 비교 시각화합니다."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(description)
        
        # 입력 섹션
        input_section = QWidget()
        input_layout = QHBoxLayout(input_section)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        # Patient ID 입력
        input_layout.addWidget(QLabel("Patient ID:"))
        self.patient_input = QLineEdit()
        self.patient_input.setPlaceholderText("예: LIDC-IDRI-0001")
        self.patient_input.setMinimumWidth(200)
        input_layout.addWidget(self.patient_input)
        
        # 다운샘플링 스텝 선택
        input_layout.addWidget(QLabel("다운샘플링:"))
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItems(["1 (전체)", "2", "3", "4", "5"])
        self.downsample_combo.setCurrentText("2")
        input_layout.addWidget(self.downsample_combo)
        
        # 불러오기 버튼
        self.load_button = QPushButton("🔍 불러오기")
        self.load_button.setMinimumHeight(35)
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2196f3;
                color: white;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                border-color: #999999;
                color: #666666;
            }
        """)
        self.load_button.clicked.connect(self.load_data)
        input_layout.addWidget(self.load_button)
        
        input_layout.addStretch()
        layout.addWidget(input_section)
        
        # 상태 메시지
        self.status_label = QLabel("Patient ID를 입력하고 '불러오기' 버튼을 클릭하세요.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            background-color: #f5f5f5;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            color: #666;
        """)
        layout.addWidget(self.status_label)
        
        # 3D 뷰어
        self.viewer = Lung3DViewer(
            info_text="피처/라벨 데이터를 불러오면 비교 시각화가 표시됩니다.",
            figure_size=(10, 8)
        )
        layout.addWidget(self.viewer)
        
        self.setLayout(layout)
        
    def load_data(self):
        """데이터 로드 및 시각화"""
        patient_id = self.patient_input.text().strip()
        
        if not patient_id:
            QMessageBox.warning(self, "입력 오류", "Patient ID를 입력해주세요.")
            return
            
        try:
            self.status_label.setText("데이터 로드 중...")
            
            # 파일 경로 구성
            feature_path = self._resolve_feature_file(patient_id)
            label_path = self._resolve_label_file(patient_id)
            
            # 데이터 로드
            features = np.load(feature_path)
            labels = np.load(label_path).astype(bool)
            
            # 데이터 검증
            if features.ndim != 2 or features.shape[1] < 3:
                raise ValueError(f"피처 데이터 형태가 잘못되었습니다: {features.shape}")
                
            if len(labels) != len(features):
                raise ValueError(f"피처와 라벨 데이터 길이가 맞지 않습니다: {len(features)} vs {len(labels)}")
            
            # 좌표 추출 (첫 3열)
            coords = features[:, :3]
            
            self.status_label.setText("예측 계산 중...")
            
            # 예측 계산
            pred_mask = self.model_manager.predict_for_organ(
                "lung", features
            )
            
            # 다운샘플링 스텝 가져오기
            downsample_text = self.downsample_combo.currentText()
            downsample_step = 1 if downsample_text.startswith("1") else int(downsample_text)
            
            # 통계 정보
            total_points = len(coords)
            pred_positive = np.sum(pred_mask)
            gt_positive = np.sum(labels)
            
            self.status_label.setText(
                f"로드 완료 - 총 점: {total_points:,}, 예측 결절: {pred_positive:,}, "
                f"실제 결절: {gt_positive:,}"
            )
            
            # 3D 뷰어 업데이트
            self.viewer.update_plot(
                coords,
                predictions=pred_mask.astype(float),
                ground_truth_mask=labels,
                downsample_step=downsample_step,
                title=f"{patient_id} 결절 비교",
                show_legend=True
            )
            
        except FileNotFoundError as e:
            QMessageBox.critical(self, "파일 오류", str(e))
            self.status_label.setText("파일을 찾을 수 없습니다.")
            
        except ValueError as e:
            QMessageBox.critical(self, "데이터 오류", str(e))
            self.status_label.setText("데이터 형식 오류가 발생했습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"예상치 못한 오류가 발생했습니다:\n{str(e)}")
            self.status_label.setText("오류가 발생했습니다.")
    
    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------
    def _resolve_feature_file(self, patient_id: str) -> Path:
        """Patient ID에 대응하는 feature 파일을 찾는다."""

        candidates = [
            self.features_dir / f"{patient_id}_features.npy",
            *sorted(self.features_dir.glob(f"{patient_id}_*_features.npy")),
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"피처 파일을 찾을 수 없습니다: {self.features_dir / (patient_id + '*')}"
        )

    def _resolve_label_file(self, patient_id: str) -> Path:
        """Patient ID에 대응하는 라벨 파일을 찾는다."""

        candidates = [
            self.labels_dir / f"{patient_id}_vertex_labels.npy",
            *sorted(self.labels_dir.glob(f"{patient_id}_*_vertex_labels.npy")),
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"라벨 파일을 찾을 수 없습니다: {self.labels_dir / (patient_id + '*')}"
        )

    def update_page(self):
        """페이지 업데이트 (필요시 호출)"""
        # 현재는 특별한 업데이트 로직이 없음
        pass
