from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Any, Dict

import numpy as np

from pipelines.common_pipeline import BaseCTPipeline
from utils.ct_loader import CTLoader
from utils.mesh_processor import MeshProcessor
from models.model_manager import ModelManager

ProgressCB = Optional[Callable[[int, str], None]]


class LungPipeline(BaseCTPipeline):
    """폐 CT를 처리해서 결절 예측을 하는 파이프라인.
    - 사전 계산된 feature가 있으면 그걸 우선 사용
    - 없으면 메쉬를 생성해서 feature를 뽑은 뒤 모델에 넣음
    """
    organ = "lung"

    def run(
        self,
        folder_path: str,
        model_manager: ModelManager,
        progress_cb: ProgressCB = None,
    ) -> Dict[str, Any]:
        folder = Path(folder_path)

        # 1) 공통 경로/ID 준비
        pid, feature_path, label_path = self.prepare_paths(folder)

        # 2) CT 로드
        self._progress(progress_cb, 10, "CT/DICOM 데이터 불러오는 중...")
        ct_loader = CTLoader()
        ct_result = ct_loader.load_and_process(str(folder))
        image = ct_result["image"]
        slices = ct_result.get("slices")

        # 3) 사전 feature 있는 경우
        if feature_path is not None:
            self._progress(progress_cb, 30, f"사전 계산 특징 발견: {feature_path.name}")
            features = np.load(str(feature_path)).astype(np.float32, copy=False)

            if features.ndim != 2 or features.shape[1] < 3:
                raise ValueError(f"feature 데이터 형태가 잘못되었습니다: {features.shape!r}")

            verts = features[:, :3].astype(np.float32, copy=False)

            self._progress(progress_cb, 50, "모델 예측 중...")
            preds, probs = model_manager.predict_segmentation(
                features, return_probabilities=True
            )

            labels = self._load_labels_if_match(label_path, verts.shape[0], progress_cb)

            prediction_accuracy = None
            if labels is not None:
                prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

            self._progress(progress_cb, 100, "폐 CT 처리 완료")

            return self.make_result_dict(
                image=image,
                verts=verts,
                predictions=preds,
                probabilities=probs,
                labels=labels,
                mesh_path=None,
                model_accuracy=getattr(model_manager, "model_accuracy", None),
                prediction_accuracy=prediction_accuracy,
                selected_folder=folder.name,
                feature_file=str(feature_path),
            )

        # 4) feature가 없으면: 메쉬 생성 → feature 추출 → 모델 예측
        self._progress(progress_cb, 40, "사전 feature가 없어 메쉬를 생성합니다...")
        mesh_proc = MeshProcessor()
        # 네 실제 MeshProcessor 시그니처에 맞춰서 tmpdir을 넘기거나 빼세요.
        mesh, mesh_path = mesh_proc.create_mesh(image, slices)

        self._progress(progress_cb, 60, "정점 특징 추출 중...")
        verts, feat, transform, dists = mesh_proc.extract_features(mesh)

        self._progress(progress_cb, 70, "모델 예측 중...")
        preds, probs = model_manager.predict_segmentation(
            feat, return_probabilities=True
        )

        labels = self._load_labels_if_match(label_path, verts.shape[0], progress_cb)

        prediction_accuracy = None
        if labels is not None and labels.size == preds.size:
            prediction_accuracy = float((preds.astype(int) == labels.astype(int)).mean())

        self._progress(progress_cb, 100, "폐 CT 처리 완료")

        return self.make_result_dict(
            image=image,
            verts=verts,
            predictions=preds,
            probabilities=probs,
            labels=labels,
            mesh_path=mesh_path,
            model_accuracy=getattr(model_manager, "model_accuracy", None),
            prediction_accuracy=prediction_accuracy,
            selected_folder=folder.name,
            feature_file=None,
        )

    def _load_labels_if_match(
        self,
        label_path: Optional[Path],
        n_verts: int,
        progress_cb: ProgressCB = None,
    ) -> Optional[np.ndarray]:
        if label_path is None:
            return None

        lab = np.load(str(label_path)).astype(bool)
        if lab.size == n_verts:
            self._progress(progress_cb, 60, f"라벨 로드 완료: {label_path.name}")
            return lab
        else:
            self._progress(
                progress_cb,
                60,
                f"라벨 길이가 맞지 않아 무시: {lab.size} vs {n_verts}",
            )
            return None
