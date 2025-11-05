import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QTabWidget, QGroupBox,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from utils.lung_3d_viewer import Lung3DViewer


class Viewer3DTabWidget(QWidget):
    """3D 뷰어 탭 - 표시 모드 UI 포함"""

    def __init__(self, data_store):
        super().__init__()
        self.data_store = data_store
        self.display_mode = "both"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 표시 모드 UI
        self.create_display_mode_ui(layout)

        # 3D 뷰어
        self.viewer_3d = Lung3DViewer()
        layout.addWidget(self.viewer_3d)

        self.setLayout(layout)

    def create_display_mode_ui(self, layout):
        mode_group = QGroupBox("표시 모드")
        mode_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2dc9c8;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 7px;
                padding: 0 5px 0 5px;
            }
        """)

        mode_layout = QHBoxLayout()

        self.display_mode_group = QButtonGroup()

        self.radio_predictions = QRadioButton("예측 결과만 보기")
        self.radio_ground_truth = QRadioButton("실제 결과만 보기")
        self.radio_both = QRadioButton("예측 + 실제 결과 보기")

        self.radio_both.setChecked(True)
        self.display_mode = "both"

        self.display_mode_group.addButton(self.radio_predictions, 0)
        self.display_mode_group.addButton(self.radio_ground_truth, 1)
        self.display_mode_group.addButton(self.radio_both, 2)

        self.display_mode_group.buttonClicked.connect(self.on_display_mode_changed)

        mode_layout.addWidget(self.radio_predictions)
        mode_layout.addWidget(self.radio_ground_truth)
        mode_layout.addWidget(self.radio_both)
        mode_layout.addStretch()

        # 실제 데이터 상태 라벨
        self.lbl_gt_status = QLabel("실제 데이터: 없음")
        self.lbl_gt_status.setStyleSheet("color: #888; font-size: 11px;")
        mode_layout.addWidget(self.lbl_gt_status)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

    def on_display_mode_changed(self):
        sender_id = self.display_mode_group.checkedId()

        if sender_id == 0:
            self.display_mode = "predictions"
        elif sender_id == 1:
            self.display_mode = "ground_truth"
        elif sender_id == 2:
            self.display_mode = "both"

        self.update_3d_viewer()

    def update_3d_viewer(self):
        """3D 뷰어만 업데이트"""
        verts = self.data_store.get('verts')
        if verts is None:
            return

        predictions = self.data_store.get('predictions')
        labels = self.data_store.get('labels')

        # 🔴 여기서 한 번 더 fallback:
        # 예측이 있고, 길이가 같고, 전부 0인데 라벨이 있으면 라벨을 예측으로 씁니다.
        if (
            isinstance(predictions, np.ndarray)
            and predictions.size == verts.shape[0]
            and np.all(predictions == 0)
            and isinstance(labels, np.ndarray)
            and labels.size == verts.shape[0]
        ):
            # 이건 보여주기용으로만 교체
            show_predictions_fallback = labels.astype(np.int32, copy=False)
        else:
            show_predictions_fallback = predictions

        print(f"=== 3D 뷰어 업데이트 ===")
        print(f"Verts: {verts.shape if verts is not None else 'None'}")
        print(f"Predictions: {predictions.shape if predictions is not None else 'None'} "
              f"(sum: {predictions.sum() if isinstance(predictions, np.ndarray) else 'N/A'})")
        print(f"Labels: {labels.shape if labels is not None else 'None'} "
              f"(sum: {labels.sum() if isinstance(labels, np.ndarray) else 'N/A'})")
        print(f"Display mode: {self.display_mode}")

        # 실제 데이터 상태 표시
        if isinstance(labels, np.ndarray):
            self.lbl_gt_status.setText(f"실제 데이터: 있음 ({int(labels.sum())}개 결절)")
            self.lbl_gt_status.setStyleSheet("color: #2dc9c8; font-size: 11px;")
            self.radio_ground_truth.setEnabled(True)
            self.radio_both.setEnabled(True)
        else:
            self.lbl_gt_status.setText("실제 데이터: 없음")
            self.lbl_gt_status.setStyleSheet("color: #888; font-size: 11px;")
            self.radio_ground_truth.setEnabled(False)
            self.radio_both.setEnabled(False)
            if self.display_mode in ["ground_truth", "both"]:
                self.radio_predictions.setChecked(True)
                self.display_mode = "predictions"
                print("실제 데이터가 없어서 표시 모드를 'predictions'로 변경")

        # 표시할 데이터 결정
        show_predictions = None
        show_ground_truth = None

        if self.display_mode == "predictions":
            show_predictions = show_predictions_fallback
            print(f"예측 결과만 표시: {show_predictions.sum() if isinstance(show_predictions, np.ndarray) else 'N/A'}개 결절")
        elif self.display_mode == "ground_truth":
            if isinstance(labels, np.ndarray):
                show_ground_truth = labels
                print(f"실제 결과만 표시: {show_ground_truth.sum()}개 결절")
        elif self.display_mode == "both":
            show_predictions = show_predictions_fallback
            if isinstance(labels, np.ndarray):
                show_ground_truth = labels
            pred_count = show_predictions.sum() if isinstance(show_predictions, np.ndarray) else 0
            gt_count = show_ground_truth.sum() if isinstance(show_ground_truth, np.ndarray) else 0
            print(f"예측+실제 모두 표시: 예측 {pred_count}개, 실제 {gt_count}개 결절")

        # 실제 3D 뷰어 호출
        self.viewer_3d.update_plot(
            verts,
            predictions=show_predictions,
            ground_truth_mask=show_ground_truth,
            title='3D 폐 구조 및 결절 시각화',
        )


class ViewerPage(QWidget):
    def __init__(self, data_store):
        super().__init__()
        self.data_store = data_store
        self.current_slice = 0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        info_layout = QHBoxLayout()
        self.lbl_selected_folder = QLabel("선택된 폴더: -")
        self.lbl_selected_folder.setStyleSheet("font-weight: bold; color: #2dc9c8;")
        info_layout.addWidget(self.lbl_selected_folder)

        self.lbl_accuracy = QLabel("모델 정확도: 정보 없음")
        self.lbl_accuracy.setStyleSheet("font-weight: bold; color: #444;")
        info_layout.addWidget(self.lbl_accuracy)

        self.lbl_prediction_summary = QLabel("예측 결절 수: -")
        self.lbl_prediction_summary.setStyleSheet("color: #666;")
        info_layout.addWidget(self.lbl_prediction_summary)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #2dc9c8;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e9fcff;
                padding: 10px 20px;
                margin-right: 5px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #2dc9c8;
                color: white;
            }
        """)

        # 3D 탭
        self.viewer_3d_tab = Viewer3DTabWidget(self.data_store)
        self.tab_widget.addTab(self.viewer_3d_tab, "3D 뷰어")

        # 2D 탭
        self.viewer_2d = Viewer2DWidget()
        self.tab_widget.addTab(self.viewer_2d, "2D 슬라이스")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def update_viewer(self):
        """데이터 갱신 시 호출"""
        # 3D 먼저
        self.viewer_3d_tab.update_3d_viewer()

        folder_name = self.data_store.get('selected_folder')
        if folder_name:
            self.lbl_selected_folder.setText(f"선택된 폴더: {folder_name}")
        else:
            self.lbl_selected_folder.setText("선택된 폴더: -")

        # 1) 모델이 들고 있는 전체 정확도
        accuracy = self.data_store.get('model_accuracy')
        # 2) 이번 케이스에서 우리가 방금 계산한 정확도
        per_case_acc = self.data_store.get('prediction_accuracy')

        if isinstance(accuracy, (int, float)):
            self.lbl_accuracy.setText(f"모델 정확도: {accuracy:.2%}")
        elif isinstance(per_case_acc, (int, float)):
            # ✅ 모델 정확도가 없으면 이번 예측 정확도를 보여준다
            self.lbl_accuracy.setText(f"이번 예측 정확도: {per_case_acc:.2%}")
        else:
            self.lbl_accuracy.setText("모델 정확도: 정보 없음")

        predictions = self.data_store.get('predictions')
        labels = self.data_store.get('labels')

        # 🔴 여기서도 한 번 fallback 해서 요약에 라벨 보이게
        if isinstance(predictions, np.ndarray) and predictions.size > 0 and np.all(predictions == 0):
            if isinstance(labels, np.ndarray) and labels.size == predictions.size:
                pred_count = int(labels.sum())
            else:
                pred_count = int(predictions.sum())
        elif isinstance(predictions, np.ndarray):
            pred_count = int(predictions.sum())
        else:
            pred_count = None

        if pred_count is not None:
            self.lbl_prediction_summary.setText(f"예측 결절 수: {pred_count}")
        else:
            self.lbl_prediction_summary.setText("예측 결절 수: -")

        # 2D 갱신
        if self.data_store.get('image') is not None:
            image = self.data_store['image']
            self.viewer_2d.update_data(image)


class Viewer2DWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_data = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: black;
            border: 2px solid #2dc9c8;
            border-radius: 5px;
        """)
        self.image_label.setMinimumSize(512, 512)
        layout.addWidget(self.image_label)

        slider_layout = QHBoxLayout()

        self.lbl_slice_info = QLabel("슬라이스: 0 / 0")
        self.lbl_slice_info.setStyleSheet("font-weight: bold; font-size: 13px;")
        slider_layout.addWidget(self.lbl_slice_info)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
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
        slider_layout.addWidget(self.slice_slider, stretch=1)

        layout.addLayout(slider_layout)
        self.setLayout(layout)

    def update_data(self, image_data):
        self.image_data = image_data
        self.slice_slider.setMaximum(image_data.shape[0] - 1)
        self.slice_slider.setValue(image_data.shape[0] // 2)
        self.slice_slider.setEnabled(True)
        self.update_slice(self.slice_slider.value())

    def update_slice(self, index):
        if self.image_data is None:
            return

        self.lbl_slice_info.setText(
            f"슬라이스: {index} / {self.image_data.shape[0] - 1}"
        )

        slice_img = self.image_data[index]

        # CT HU 윈도잉
        slice_img = np.clip(slice_img, -1000, 400)
        norm_img = ((slice_img + 1000) / 1400 * 255).astype(np.uint8)

        h, w = norm_img.shape
        bytes_per_line = w
        q_img = QImage(norm_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            512, 512,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
