# pipelines/common_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np

from utils.config import FEATURES_DIR, LABELS_DIR
from utils.dicom_processor import DICOMProcessor

# 진행률 콜백 타입: (퍼센트:int, 메시지:str) -> None
ProgressCB = Optional[Callable[[int, str], None]]


def _extract_id_from_folder(folder: Path) -> str:
    """
    폴더 이름에서 케이스/환자 ID를 추출하는 기본 규칙.
    예) 'CT-0001' -> '0001'
        '0001'     -> '0001'
    프로젝트 전체에서 규칙이 하나로 맞아야 하니,
    필요하면 여기만 바꿔주면 된다.
    """
    name = folder.name
    if "-" in name:
        return name.split("-")[-1]
    return name


class BaseCTPipeline:
    """
    CT/DICOM을 기반으로 하는 파이프라인들이 공통으로 사용하는 베이스 클래스.

    이 클래스가 해주는 일:
    1. 폴더에서 ID 추출
    2. ID로 사전 계산 feature / label 파일 경로 찾기
    3. 진행률 콜백을 안전하게 호출
    4. DICOM을 실제로 로드하는 기본 메서드 제공
    5. 최종 결과를 dict로 통일하는 메서드 제공

    LungPipeline, LiverPipeline 같은 파이프라인들은
    이 클래스를 상속해서 run(...) 안에 자기 장기별 로직만 넣으면 된다.
    """

    # 하위 클래스에서 organ = "lung" / "liver" 처럼 덮어써도 됨
    organ: str = "ct"

    # ------------------------------------------------------------------
    # 1) ID, feature, label 경로 준비
    # ------------------------------------------------------------------
    def prepare_paths(self, folder: Path):
        """
        폴더에서 ID를 뽑고, 그 ID로 feature/label 파일을 찾는다.

        반환:
            pid: str
            feature_path: Path | None
            label_path: Path | None
        """
        pid = _extract_id_from_folder(folder)

        # <ID>_features.npy
        feature_path = FEATURES_DIR / f"{pid}_features.npy"
        if not feature_path.exists():
            feature_path = None

        # <ID>_vertex_labels.npy
        label_path = LABELS_DIR / f"{pid}_vertex_labels.npy"
        if not label_path.exists():
            label_path = None

        return pid, feature_path, label_path

    # ------------------------------------------------------------------
    # 2) 진행률 호출
    # ------------------------------------------------------------------
    def _progress(self, cb: ProgressCB, pct: int, msg: str):
        """
        콜백이 있으면 호출하고, 없으면 그냥 무시한다.
        이렇게 해두면 UI가 없는 환경에서도 파이프라인을 돌릴 수 있다.
        """
        if cb:
            cb(pct, msg)

    # ------------------------------------------------------------------
    # 3) CT/DICOM 로드
    # ------------------------------------------------------------------
    def load_ct(self, folder: Path, progress_cb: ProgressCB = None) -> Dict[str, Any]:
        """
        DICOM 폴더를 로드해서 image/slices 등을 반환한다.
        원래 네 코드에서 쓰던 DICOMProcessor를 그대로 쓴다.
        장기에 따라 다르게 읽어야 하면 하위 클래스에서 이 메서드를 오버라이드하면 된다.
        """
        self._progress(progress_cb, 10, "DICOM 파일 로드 중...")
        dcm_proc = DICOMProcessor()
        dcm_result = dcm_proc.load_and_process(str(folder))
        # dcm_result는 원래 네 코드처럼 {"image": ..., "slices": ...} 형태라고 가정
        return dcm_result

    # ------------------------------------------------------------------
    # 4) 라벨 로드 (공통)
    # ------------------------------------------------------------------
    def load_labels_if_match(
        self,
        label_path: Optional[Path],
        n_verts: int,
        progress_cb: ProgressCB = None,
    ) -> Optional[np.ndarray]:
        """
        라벨 파일이 있고, 정점 수와 길이가 맞으면 bool array로 반환.
        길이가 다르면 None을 반환하고 로그만 남긴다.
        """
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

    # ------------------------------------------------------------------
    # 5) 결과 포맷
    # ------------------------------------------------------------------
    def make_result_dict(self, **kwargs) -> Dict[str, Any]:
        """
        파이프라인 실행 결과를 dict로 포맷할 때 사용.
        pages 쪽에서 이 dict를 그대로 data_store에 넣어 쓰게 되므로
        여기서는 별도로 가공하지 않고 그대로 돌려준다.

        필요하면 여기서 공통 키를 강제할 수도 있다.
        """
        return kwargs
