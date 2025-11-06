# test.py
from models import ModelLoader, ModelManager
import numpy as np

def main():
    print("=== ModelLoader 테스트 시작 ===")
    loader = ModelLoader()
    loaded = loader.load()
    print("로드 결과 키:", loaded.keys())

    manager = ModelManager(loaded)

    # 가짜 feature 16개, 채널 7개
    features = np.random.rand(16, 7).astype("float32")

    print("=== ModelManager 예측 테스트 ===")
    try:
        preds, probs = manager.predict_segmentation(features, return_probabilities=True)
        print("예측 성공")
        print("preds.shape:", preds.shape)
        print("probs.shape:", probs.shape)
    except Exception as e:
        # 여기서 에러 내용을 보자
        import traceback
        print("예측 중 오류 발생:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
