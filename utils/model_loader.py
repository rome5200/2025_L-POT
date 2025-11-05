from __future__ import annotations

from pathlib import Path
import numpy as np

# PyTorch 안전 임포트
try:
    from torch.serialization import add_safe_globals
    import torch
    from models.spiralnet import (
        HybridSpiralNetPointNetTransformer,
        ImprovedSpiralNet,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch 임포트 실패: {e}")
    TORCH_AVAILABLE = False


class ModelManager:
    """모델 로딩 및 추론 관리"""
    
    def __init__(self, base_path: Path | None = None):
        """Create a new model manager.

        Parameters
        ----------
        base_path:
            Optional base directory that contains the weight and index files.
            When ``None`` the project root is used.  This makes the loader work
            regardless of the runtime environment instead of relying on a
            developer specific absolute Windows path.
        """

        self.torch_available = TORCH_AVAILABLE
        self.spiralnet_model = None
        self.threshold = 0.5
        self.expected_vertex_count: int | None = None
        self.model_accuracy: float | None = None

        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            project_root = Path(base_path) if base_path is not None else Path(__file__).resolve().parents[1]
            
            # 경로 설정
            self.SPIRAL_PATH = project_root / "models/spiral_9.npy"
            self.SPIRALNET_WEIGHT = project_root / "new_2.pth"
            
            self._load_models()
        else:
            print("PyTorch가 사용할 수 없습니다. 더미 예측을 사용합니다.")
    
    def _load_models(self):
        """모델 로드"""
        try:
            if not self.SPIRAL_PATH.exists():
                raise FileNotFoundError(f"Spiral index file not found: {self.SPIRAL_PATH}")

            if not self.SPIRALNET_WEIGHT.exists():
                raise FileNotFoundError(f"Model weight file not found: {self.SPIRALNET_WEIGHT}")

            # Spiral indices 로드
            spiral_all = np.load(self.SPIRAL_PATH, allow_pickle=True)
            self.expected_vertex_count = int(spiral_all.shape[0])
            num_blocks = 4
            spiral_idx = [torch.from_numpy(spiral_all).long().to(self.device) 
                         for _ in range(num_blocks)]
            
            # SpiralNet 모델
            spiralnet = ImprovedSpiralNet(
                in_channels=7,
                hidden_channels=256,
                out_channels=1,
                num_blocks=num_blocks,
                spiral_indices=spiral_idx,
                dropout=0.3,
                use_se=True
            ).to(self.device)
            
            self.spiralnet_model = HybridSpiralNetPointNetTransformer(
                spiralnet=spiralnet,
                in_channels=7,
                hidden_dim=256,
                out_dim=1,
                dropout=0.3,
                transformer_heads=4,
                transformer_blocks=1
            ).to(self.device)
            
            # 가중치 로드 - numpy.dtype을 안전한 전역 객체로 추가
            add_safe_globals([np.core.multiarray.scalar, np.dtype])

            # 방법 1: weights_only=False 사용 (신뢰할 수 있는 파일인 경우)
            ckpt = torch.load(self.SPIRALNET_WEIGHT, map_location=self.device, weights_only=False)
            
            # 방법 2 (대안): weights_only=True를 유지하고 싶다면 아래 코드 사용
            # ckpt = torch.load(self.SPIRALNET_WEIGHT, map_location=self.device, weights_only=True)
            
            state = ckpt.get('model_state_dict', ckpt)  # plain state_dict일 수도 있으니 호환 처리
            self.spiralnet_model.load_state_dict(state, strict=True)
            if isinstance(ckpt, dict):
                self.threshold = float(ckpt.get('threshold', 0.5))
                accuracy = ckpt.get('accuracy')
                if accuracy is not None:
                    try:
                        self.model_accuracy = float(accuracy)
                    except (TypeError, ValueError):
                        self.model_accuracy = None
            else:
                self.threshold = 0.5

            self.spiralnet_model.eval()
            self.spiral_idx = spiral_idx
            print(f"모델 로드 완료 (Device: {self.device})")
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise
    
    def _infer_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Return per-vertex probabilities for the given features."""
        if not self.torch_available or self.spiralnet_model is None:
            print("PyTorch 모델을 사용할 수 없습니다. 더미 예측을 생성합니다.")
            return self._generate_dummy_probabilities(features)

        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            spiral_idx_batch = [idx.unsqueeze(0) for idx in self.spiral_idx]

            logits = self.spiralnet_model(features_tensor, spiral_idx_batch)
            logits = logits.detach().cpu().numpy()

            if logits.ndim == 2:
                logits = logits[0]

            # Sigmoid
            probs = 1 / (1 + np.exp(-logits))

        return probs.astype(np.float32, copy=False)

    def predict_segmentation(self, features: np.ndarray, *, return_probabilities: bool = False):
        """세그멘테이션 예측"""
        probs = self._infer_probabilities(features)
        predictions = probs > self.threshold

        if return_probabilities:
            return predictions, probs

        return predictions

    def _generate_dummy_probabilities(self, features):
        """더미 확률 데이터 생성 (테스트용)"""
        n_vertices = features.shape[0]

        probs = np.zeros(n_vertices, dtype=np.float32)
        if n_vertices:
            center_indices = np.random.choice(n_vertices, size=min(50, max(1, n_vertices // 20)), replace=False)
            probs[center_indices] = 0.85

        print(f"더미 예측 생성: {(probs > self.threshold).sum()} / {n_vertices} 정점이 결절로 예측됨")
        return probs

    def predict_from_feature_file(self, feature_file: Path):
        """Load a precomputed feature file and run inference."""

        feature_path = Path(feature_file)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        try:
            features = np.load(feature_path)
        except Exception as exc:
            raise ValueError(f"Failed to load feature file: {exc}") from exc

        if not isinstance(features, np.ndarray) or features.ndim != 2 or features.shape[1] < 3:
            raise ValueError("Feature array must be a 2D numpy array with at least three columns.")

        features = np.asarray(features, dtype=np.float32)
        predictions, probabilities = self.predict_segmentation(features, return_probabilities=True)
        verts = features[:, :3].astype(np.float32, copy=False)

        return {
            "features": features,
            "verts": verts,
            "predictions": predictions,
            "probabilities": probabilities,
            "accuracy": self.model_accuracy,
        }
