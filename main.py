import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QStackedWidget, QPushButton, QLabel,
                              QFileDialog, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pages.upload_page import UploadPage
from pages.viewer_page import ViewerPage
from pages.feature_label_viewer_page import FeatureLabelViewerPage
from utils.model_loader import ModelManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT-Dicom 기반 3D 시각화 및 결절 탐지 시스템")
        self.setGeometry(100, 100, 1400, 900)
        
        # 모델 매니저 초기화
        self.model_manager = ModelManager()
        
        # 데이터 저장용
        self.current_data = {
            'image': None,
            'verts': None,
            'labels': None,
            'predictions': None,
            'probabilities': None,
            'mesh_path': None,
            'model_accuracy': None,
            'feature_file': None,
            'selected_folder': None,
        }
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 헤더
        header = QLabel("CT-Dicom 기반 3D 시각화 및 결절 탐지 시스템")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_font = QFont()
        header_font.setPointSize(18)
        header_font.setBold(True)
        header.setFont(header_font)
        main_layout.addWidget(header)
        
        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        
        self.btn_upload = QPushButton("파일 업로드")
        self.btn_viewer = QPushButton("CT 뷰어")
        self.btn_feature_viewer = QPushButton("📌 피처/라벨 뷰어")
        
        for btn in [self.btn_upload, self.btn_viewer, self.btn_feature_viewer]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e9fcff;
                    border: 2px solid #2dc9c8;
                    border-radius: 8px;
                    padding: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2dc9c8;
                    color: white;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    border-color: #999999;
                    color: #666666;
                }
            """)
            nav_layout.addWidget(btn)
        
        main_layout.addLayout(nav_layout)
        
        # 스택 위젯 (페이지 전환용)
        self.stacked_widget = QStackedWidget()
        
        # 페이지 생성
        self.upload_page = UploadPage(self.model_manager, self.current_data)
        self.viewer_page = ViewerPage(self.current_data)
        self.feature_page = FeatureLabelViewerPage(self.model_manager)
        
        self.stacked_widget.addWidget(self.upload_page)
        self.stacked_widget.addWidget(self.viewer_page)
        self.stacked_widget.addWidget(self.feature_page)
        
        main_layout.addWidget(self.stacked_widget)
        
        # 하단 안내문
        footer = QLabel(
            "본 시스템은 연구용으로 제작된 시스템이며,\n"
            "정확한 진단은 반드시 전문 의료진의 판독을 참고하시기 바랍니다."
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #888; font-size: 12px; margin-top: 20px;")
        main_layout.addWidget(footer)
        
        # 버튼 연결
        self.btn_upload.clicked.connect(lambda: self.switch_page(0))
        self.btn_viewer.clicked.connect(lambda: self.switch_page(1))
        self.btn_feature_viewer.clicked.connect(lambda: self.switch_page(2))
        
        # 페이지 간 신호 연결
        self.upload_page.processing_completed.connect(self.on_processing_completed)
        
        # 초기 상태
        self.btn_viewer.setEnabled(False)
        
    def switch_page(self, index):
        """페이지 전환"""
        self.stacked_widget.setCurrentIndex(index)
        
        # 페이지 업데이트
        if index == 1:  # 뷰어 페이지
            self.viewer_page.update_viewer()
        elif index == 2:  # 피처/라벨 뷰어 페이지
            self.feature_page.update_page()
    
    def on_processing_completed(self):
        """처리 완료 시 호출"""
        self.btn_viewer.setEnabled(True)
        
        # 자동으로 뷰어 페이지로 이동
        self.switch_page(1)


def main():
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
