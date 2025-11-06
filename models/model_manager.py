# models/model_manager.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    """
    ModelLoader가 로드해 준 모델/메타데이터를 받아서
    실제 추론을 담당하는 매니저.

    - segmentation 예측
    - feature 파일을 바로 넣어서 예측
    """

    def __init__(self, loaded: Dict[str, Any]):
        """
        Parameters
        ----------
        loaded: dict
            models.model_loader.ModelLoader.load() 가 돌려준 dict
        """
        self.model = loaded.get("model")
        self.device = loaded.get("device")
        self.spiral_idx = loaded.get("spiral_idx")
        self.threshold: float = loaded.get("threshold", 0.5)
        self.model_accuracy: Optional[float] = loaded.get("model_accuracy")
        self.expected_vertex_count: Optional[int] = loaded.get("expected_vertex_count")

        # torch가 있어야 하고, 모델도 있어야 진짜 추론 가능
        self.torch_available = (
            TORCH_AVAILABLE and self.model is not None and self.device is not None
        )

    # ------------------------------------------------------------------
    # 내부: 확률 예측
    # ------------------------------------------------------------------
    def _infer_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        주어진 정점 feature에 대해 모델이 출력한 per-vertex 확률을 반환한다.
        모델이 준비되지 않았으면 RuntimeError를 던진다.
        """
        if not self.torch_available:
            raise RuntimeError("모델이 로드되지 않았거나 PyTorch를 사용할 수 없습니다.")

        self.model.eval()

        # (1, N, C)로 맞춰서 모델에 넣음
        with torch.no_grad():
            x = (
                torch.tensor(features, dtype=torch.float32)
                .unsqueeze(0)  # batch 차원
                .to(self.device)
            )
            # spiral idx도 batch 차원 맞춰서 전달
            spiral_idx_batch = [idx.unsqueeze(0) for idx in self.spiral_idx]

            logits = self.model(x, spiral_idx_batch)
            logits = logits.detach().cpu().numpy()

        # (1, N) 형태면 (N,)로 펴기
        if logits.ndim == 2:
            logits = logits[0]

        # 시그모이드로 확률화
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # 외부: 세그멘테이션
    # ------------------------------------------------------------------
    def predict_segmentation(
        self,
        features: np.ndarray,
        *,
        return_probabilities: bool = False,
    ):
        """
        세그멘테이션 예측
        features: (N, C) numpy array

        return_probabilities=True 이면 (preds, probs) 튜플을 반환
        """
        probs = self._infer_probabilities(features)
        preds = probs > self.threshold

        if return_probabilities:
            return preds, probs
        return preds

    # ------------------------------------------------------------------
    # feature 파일에서 바로 예측
    # ------------------------------------------------------------------
    def predict_from_feature_file(self, feature_file: Path | str) -> Dict[str, Any]:
        """
        사전 계산된 feature npy 파일을 바로 읽어서 예측까지 한 번에 수행한다.
        원래 네가 쓰던 predict_from_feature_file 기능을 그대로 옮긴 것.
        """
        feature_path = Path(feature_file)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        try:
            features = np.load(feature_path)
        except Exception as exc:
            raise ValueError(f"Failed to load feature file: {exc}") from exc

        # 최소한의 유효성 검사
        if not isinstance(features, np.ndarray) or features.ndim != 2 or features.shape[1] < 3:
            raise ValueError("Feature array must be 2D ndarray with at least 3 columns.")

        features = np.asarray(features, dtype=np.float32)

        preds, probs = self.predict_segmentation(features, return_probabilities=True)
        verts = features[:, :3].astype(np.float32, copy=False)

        return {
            "features": features,
            "verts": verts,
            "predictions": preds,
            "probabilities": probs,
            "accuracy": self.model_accuracy,
        }
