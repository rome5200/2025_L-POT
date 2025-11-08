# pipelines/common_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple
import tempfile

import numpy as np

from utils import DICOMProcessor, MeshProcessor  # lazy import 구조에 맞춤 :contentReference[oaicite:4]{index=4}
from utils.config import FEATURES_DIR, FEATURES_DIRS, LABELS_DIR, LABELS_DIRS

ProgressCB = Optional[Callable[[int, str], None]]


class BaseCTPipeline:
    """
    DICOM → (이미지, 슬라이스) → feature/label 파일 경로 찾는 공통 파이프라인.
    장기 구분은 여기엔 없음.
    """

    def __init__(self) -> None:
        self.features_dir: Path = FEATURES_DIR
        self.labels_dir: Path = LABELS_DIR

    # 공통 progress 헬퍼
    def _progress(self, cb: ProgressCB, pct: int, msg: str) -> None:
        if cb is not None:
            cb(int(pct), msg)

    def _extract_id_from_folder(self, folder: Path) -> str:
        """
        폴더 이름에서 케이스 ID를 만든다.
        이제는 이 ID가 곧 파일 이름이 될 거라서 접미사 안 붙인다.
        예) '2025-CT-001' -> '2025-CT-001'
        """
        return folder.name

    def _feature_path_for(self, pid: str) -> Path:
        return self.features_dir / f"{pid}_features.npy"

    def _label_path_for(self, pid: str) -> Path:
        return self.labels_dir / f"{pid}_vertex_labels.npy"

    def prepare_paths(self, folder: Path) -> Tuple[str, Path, Path]:
        pid = self._extract_id_from_folder(folder)
        return pid, self._feature_path_for(pid), self._label_path_for(pid)

    def load_ct(self, folder: Path, progress_cb: ProgressCB = None) -> Dict[str, Any]:
        self._progress(progress_cb, 10, "DICOM 불러오는 중...")
        dcm = DICOMProcessor()
        ct = dcm.load_and_process(str(folder))  # image, mask, spacing, slices 전부 줌 :contentReference[oaicite:6]{index=6}
        self._progress(progress_cb, 25, "DICOM 로드 완료")
        return ct

    def _load_labels_if_valid(
        self,
        label_path: Path,
        n_verts: int,
        progress_cb: ProgressCB = None,
    ) -> Optional[np.ndarray]:
        if not label_path.exists():
            return None
        labels = np.load(str(label_path)).astype(bool)
        if labels.size == n_verts:
            self._progress(progress_cb, 70, f"정답 라벨 로드: {label_path.name}")
            return labels
        self._progress(progress_cb, 70, "라벨 길이가 안 맞아 무시합니다.")
        return None

    def make_result(
        self,
        *,
        image: np.ndarray,
        verts: Optional[np.ndarray],
        predictions: Optional[np.ndarray],
        probabilities: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        mesh_path: Optional[str],
        model_accuracy: Optional[float],
        prediction_accuracy: Optional[float],
        selected_folder: str,
        feature_file: Optional[str],
        organ: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "image": image,
            "verts": verts,
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": labels,
            "mesh_path": mesh_path,
            "model_accuracy": model_accuracy,
            "prediction_accuracy": prediction_accuracy,
            "selected_folder": selected_folder,
            "feature_file": feature_file,
            "organ": organ,
        }


class OrganCTPipeline(BaseCTPipeline):
    """
    장기 이름만 바꿔서 CT 파이프라인을 돌리는 래퍼.
    """

    def __init__(self, organ: str):
        super().__init__()
        self.organ = organ
        self.features_dir = FEATURES_DIRS[organ]
        self.labels_dir = LABELS_DIRS[organ]

    def run(
        self,
        folder_path: str,
        model_manager,
        progress_cb: ProgressCB = None,
    ) -> Dict[str, Any]:
        folder = Path(folder_path)

        # 1) ID 및 파일 경로 준비
        pid, feature_path, label_path = self.prepare_paths(folder)

        # 2) CT 로드
        ct = self.load_ct(folder, progress_cb)
        image = ct["image"]
        slices = ct["slices"]

        # 3) 사전 계산 feature가 있으면 그걸로 바로 예측
        if feature_path.exists():
            self._progress(progress_cb, 40, f"사전 feature 사용: {feature_path.name}")
            features = np.load(str(feature_path)).astype(np.float32, copy=False)
            verts = features[:, :3].astype(np.float32, copy=False)

            self._progress(progress_cb, 60, "모델 예측 중...")
            preds, probs = model_manager.predict_for_organ(
                self.organ,
                features,
                return_probabilities=True,
            )

            labels = self._load_labels_if_valid(label_path, verts.shape[0], progress_cb)

            pred_acc = None
            if labels is not None:
                pred_acc = float((preds.astype(int) == labels.astype(int)).mean())

            self._progress(progress_cb, 100, "완료")

            return self.make_result(
                image=image,
                verts=verts,
                predictions=preds,
                probabilities=probs,
                labels=labels,
                mesh_path=None,
                model_accuracy=getattr(model_manager, "model_accuracy", None),
                prediction_accuracy=pred_acc,
                selected_folder=folder.name,
                feature_file=str(feature_path),
                organ=self.organ,
            )

        # 4) feature가 없으면 → 메쉬 생성 → feature 추출 → 예측
        self._progress(progress_cb, 40, "메쉬 생성 중...")
        mesh_proc = MeshProcessor()  # 너네 utils 기준으로 맞춤 :contentReference[oaicite:7]{index=7}
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh, mesh_path = mesh_proc.create_mesh(image, slices, tmpdir=tmpdir)

        self._progress(progress_cb, 55, "정점 특징 추출 중...")
        verts, feat, transform, dists = mesh_proc.extract_features(mesh)

        self._progress(progress_cb, 70, "모델 예측 중...")
        preds, probs = model_manager.predict_for_organ(
            self.organ,
            feat,
            return_probabilities=True,
        )

        labels = self._load_labels_if_valid(label_path, verts.shape[0], progress_cb)

        pred_acc = None
        if labels is not None and labels.size == preds.size:
            pred_acc = float((preds.astype(int) == labels.astype(int)).mean())

        self._progress(progress_cb, 100, "완료")

        return self.make_result(
            image=image,
            verts=verts,
            predictions=preds,
            probabilities=probs,
            labels=labels,
            mesh_path=mesh_path,
            model_accuracy=getattr(model_manager, "model_accuracy", None),
            prediction_accuracy=pred_acc,
            selected_folder=folder.name,
            feature_file=None,
            organ=self.organ,
        )
