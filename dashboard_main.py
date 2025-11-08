# dashboard_main.py

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow

# ① 대시보드 페이지
from pages.dashboard_page import DashboardPage

# ② 모델 관련
from models.model_loader import ModelLoader   # 실제 모델/스파이럴 로더
from models.model_manager import ModelManager


def load_model_manager() -> ModelManager:
    """
    모델을 로드해서 ModelManager에 넘겨준다.
    models/model_loader.py가 lung/liver 둘 다 로드하도록 되어 있으므로
    실패 fallback도 그 키 구조에 맞춰준다.
    """
    base_path = Path(__file__).resolve().parent

    try:
        loader = ModelLoader(base_path=base_path)
        loaded_dict = loader.load()
        return ModelManager(loaded_dict)
    except Exception as e:
        # 모델 파일이 없거나 torch가 없거나 경로가 안 맞는 경우
        print(f"[dashboard_main] 모델 로드 실패: {e}")

        # 현재 model_manager는 lung/liver, spiral_idx_lung, spiral_idx_liver 등을 기대할 수 있으므로
        # 그 구조에 최대한 맞춰서 빈 dict를 만든다.
        fallback = {
            "device": None,
            "expected_vertex_count": None,
            # spiral
            "spiral_idx": None,
            "spiral_idx_lung": None,
            "spiral_idx_liver": None,
            # models
            "lung_model": None,
            "liver_model": None,
            # thresholds / metrics
            "threshold": 0.5,
            "model_accuracy": None,
            "liver_threshold": 0.5,
            "liver_accuracy": None,
        }
        return ModelManager(fallback)


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 실제 모델 로드
        model_manager = load_model_manager()

        # 파이프라인/뷰어와 공유할 저장소
        data_store = {}

        # 대시보드 페이지 붙이기
        dashboard = DashboardPage(model_manager=model_manager, data_store=data_store)
        self.setCentralWidget(dashboard)

        self.setWindowTitle("L-POT : Lung / Liver Prediction Tool")
        self.resize(1400, 800)


def main():
    # 일부 환경에서 MKL 충돌 때문에 필요할 수 있음
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    app = QApplication(sys.argv)

    win = DashboardWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
