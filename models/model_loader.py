# models/model_loader.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from utils.helper import resource_path  # 프로젝트 루트 기준 경로 가져오기 :contentReference[oaicite:5]{index=5}

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
    모델과 필요한 부가 리소스(spiral 인덱스, threshold, accuracy)를 로드해서
    models/model_manager.py 가 바로 쓸 수 있는 dict로 돌려준다. :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, base_path: Path | None = None):
        self.torch_available = TORCH_AVAILABLE
        self.base_path = (
            Path(base_path)
            if base_path is not None
            else Path(__file__).resolve().parents[1]
        )

        # 공통 리소스
        self.SPIRAL_PATH = resource_path("models/spiral_9.npy")
        # 폐/간 가중치
        self.LUNG_WEIGHT_PATH = resource_path("models/new_2.pth")
        self.LIVER_WEIGHT_PATH = resource_path("models/mask_merged_ver4.pth")

    def _build_backbone(self, spiral_idx_list, device):
        num_blocks = len(spiral_idx_list)

        spiralnet = ImprovedSpiralNet(
            in_channels=7,
            hidden_channels=256,
            out_channels=1,
            num_blocks=num_blocks,
            spiral_indices=spiral_idx_list,
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

        return model

    def _load_weight(self, model, weight_path: Path, device):
        add_safe_globals([np.core.multiarray.scalar, np.dtype])
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)

        threshold = 0.5
        accuracy = None
        if isinstance(ckpt, dict):
            threshold = float(ckpt.get("threshold", 0.5))
            acc = ckpt.get("accuracy")
            if acc is not None:
                try:
                    accuracy = float(acc)
                except (TypeError, ValueError):
                    accuracy = None

        model.eval()
        return model, threshold, accuracy

    def load(self):
        if not self.torch_available:
            print("PyTorch가 사용할 수 없습니다. 더미 예측만 반환합니다.")
            return {
                "device": None,
                "spiral_idx": None,
                "expected_vertex_count": None,
                "lung_model": None,
                "liver_model": None,
                "threshold": 0.5,
                "liver_threshold": 0.5,
                "model_accuracy": None,
                "liver_accuracy": None,
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.SPIRAL_PATH.exists():
            raise FileNotFoundError(f"Spiral index file not found: {self.SPIRAL_PATH}")

        spiral_all = np.load(self.SPIRAL_PATH, allow_pickle=True)
        expected_vertex_count = int(spiral_all.shape[0])

        # spiral 리스트 만들기
        num_blocks = 4
        spiral_idx_list = [
            torch.from_numpy(spiral_all).long().to(device) for _ in range(num_blocks)
        ]

        # 1) 폐 모델
        if not self.LUNG_WEIGHT_PATH.exists():
            raise FileNotFoundError(f"Lung model weight file not found: {self.LUNG_WEIGHT_PATH}")
        lung_model = self._build_backbone(spiral_idx_list, device)
        lung_model, lung_thr, lung_acc = self._load_weight(
            lung_model, self.LUNG_WEIGHT_PATH, device
        )

        # 2) 간 모델 (있으면)
        liver_model = None
        liver_thr = lung_thr
        liver_acc = None
        if self.LIVER_WEIGHT_PATH.exists():
            liver_model = self._build_backbone(spiral_idx_list, device)
            liver_model, liver_thr, liver_acc = self._load_weight(
                liver_model, self.LIVER_WEIGHT_PATH, device
            )
        else:
            print("[models.model_loader] 간 가중치가 없어서 폐 모델만 로드했습니다.")

        print(f"모델 로드 완료 (device={device})")

        return {
            "device": device,
            "expected_vertex_count": expected_vertex_count,
            # 공통 spiral + 장기별 spiral (지금은 동일한 걸 넣음)
            "spiral_idx": spiral_idx_list,
            "spiral_idx_lung": spiral_idx_list,
            "spiral_idx_liver": spiral_idx_list,
            # 모델
            "lung_model": lung_model,
            "liver_model": liver_model,
            # threshold / accuracy
            "threshold": lung_thr,
            "model_accuracy": lung_acc,
            "liver_threshold": liver_thr,
            "liver_accuracy": liver_acc,
        }
