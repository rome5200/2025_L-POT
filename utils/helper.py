# utils/path_helper.py 같은 데에 두면 좋음
import sys
from pathlib import Path

def resource_path(relative: str) -> Path:
    """exe로 묶였을 때와 개발 중일 때 모두에서 쓸 수 있는 리소스 경로 얻기"""
    if hasattr(sys, "_MEIPASS"):
        # pyinstaller가 임시로 풀어준 디렉터리
        base = Path(sys._MEIPASS)
    else:
        # 개발 중: 프로젝트 루트 기준
        base = Path(__file__).resolve().parents[1]
    return base / relative
