# models/model_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from utils import resource_path

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


class ModelLoader:
    """
    모델과 필요한 부가 리소스(spiral 인덱스, threshold, accuracy)를 로드하는 책임만 가진다.
    실제 추론은 여기서 하지 않는다.
    """

    def __init__(self, base_path: Path | None = None):
        self.torch_available = TORCH_AVAILABLE
        self.base_path = Path(base_path) if base_path is not None else Path(__file__).resolve().parents[1]

        # 기본 경로 설정 (원래 코드 유지)
        self.SPIRAL_PATH = resource_path("models/spiral_9.npy")
        self.WEIGHT_PATH = resource_path("models/new_2.pth")

    def load(self):
        """
        모델, spiral 인덱스, 메타데이터를 로드해서 하나의 dict로 반환한다.
        추론은 이걸 받은 쪽에서 한다.
        """
        if not self.torch_available:
            print("PyTorch가 사용할 수 없습니다. 더미 예측만 가능합니다.")
            return {
                "model": None,
                "device": None,
                "spiral_idx": None,
                "threshold": 0.5,
                "model_accuracy": None,
                "expected_vertex_count": None,
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.SPIRAL_PATH.exists():
            raise FileNotFoundError(f"Spiral index file not found: {self.SPIRAL_PATH}")

        if not self.WEIGHT_PATH.exists():
            raise FileNotFoundError(f"Model weight file not found: {self.WEIGHT_PATH}")

        # 1) spiral 인덱스 로드
        spiral_all = np.load(self.SPIRAL_PATH, allow_pickle=True)
        expected_vertex_count = int(spiral_all.shape[0])

        num_blocks = 4
        spiral_idx = [torch.from_numpy(spiral_all).long().to(device) for _ in range(num_blocks)]

        # 2) 모델 구성 (원래 코드 그대로)
        spiralnet = ImprovedSpiralNet(
            in_channels=7,
            hidden_channels=256,
            out_channels=1,
            num_blocks=num_blocks,
            spiral_indices=spiral_idx,
            dropout=0.3,
            use_se=True,
        ).to(device)

        model = HybridSpiralNetPointNetTransformer(
            spiralnet=spiralnet,
            in_channels=7,
            hidden_dim=256,
            out_dim=1,
            dropout=0.3,
            transformer_heads=4,
            transformer_blocks=1,
        ).to(device)

        # 3) 가중치 로드
        add_safe_globals([np.core.multiarray.scalar, np.dtype])
        ckpt = torch.load(self.WEIGHT_PATH, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)

        threshold = 0.5
        model_accuracy = None
        if isinstance(ckpt, dict):
            threshold = float(ckpt.get("threshold", 0.5))
            accuracy = ckpt.get("accuracy")
            if accuracy is not None:
                try:
                    model_accuracy = float(accuracy)
                except (TypeError, ValueError):
                    model_accuracy = None

        model.eval()
        print(f"모델 로드 완료 (Device: {device})")

        return {
            "model": model,
            "device": device,
            "spiral_idx": spiral_idx,
            "threshold": threshold,
            "model_accuracy": model_accuracy,
            "expected_vertex_count": expected_vertex_count,
        }
