"""
EAR (Eye Aspect Ratio) 계산 + 사용자별 상대 EAR 캘리브레이션.

68점 랜드마크(dlib/300W 표준 규약) 기준 눈 인덱스:
  왼쪽 눈:  36, 37, 38, 39, 40, 41
  오른쪽 눈: 42, 43, 44, 45, 46, 47

EAR 공식 (Soukupova & Cech, 2016):
  EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
  여기서 p1..p6은 눈 주위 6점 (바깥끝, 위2점, 안끝, 아래2점)
눈을 뜨면 세로 거리가 커서 EAR이 크고, 감으면 0에 가까워진다.
"""

import numpy as np
from collections import deque

# 68점 규약의 눈 인덱스
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]


def _eye_aspect_ratio(pts6):
    """pts6: (6,2) 배열. 눈 6점. 표준 순서 [p1,p2,p3,p4,p5,p6]."""
    p1, p2, p3, p4, p5, p6 = pts6
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def compute_ear(landmarks68):
    """
    landmarks68: (68,2) 배열, 이미지 좌표.
    반환: 양쪽 눈 EAR 평균.
    """
    left = _eye_aspect_ratio(landmarks68[LEFT_EYE])
    right = _eye_aspect_ratio(landmarks68[RIGHT_EYE])
    return (left + right) / 2.0


class EyeStateJudge:
    """
    상대 EAR 방식으로 눈 감김을 판정.

    계획서 4.1: 시작 후 약 3초간 눈 뜬 정상 EAR을 측정,
    그 중앙값을 기준값으로. 절대 임계값 대신 상대값을 써서
    사용자별 눈 크기 차이에 따른 오탐을 줄인다.

    Relative EAR = 현재 EAR / 기준 EAR
    이 값이 close_ratio 미만이면 눈 감김으로 판정.
    """

    def __init__(self, calib_sec=3.0, close_ratio=0.75):
        self.calib_sec = calib_sec
        self.close_ratio = close_ratio
        self._calib_values = []      # (ear,) 캘리브레이션 구간 수집
        self._calib_start_ts = None
        self.baseline = None         # 기준 EAR (중앙값)

    def is_calibrated(self):
        return self.baseline is not None

    def update(self, timestamp, ear):
        """
        반환: dict
            calibrating   : 아직 캘리브레이션 중인지
            baseline      : 기준 EAR (없으면 None)
            relative_ear  : 상대 EAR (캘리브 전이면 None)
            eye_closed    : 눈 감김 여부 (캘리브 전이면 False)
        """
        if self._calib_start_ts is None:
            self._calib_start_ts = timestamp

        # 캘리브레이션 구간
        if self.baseline is None:
            elapsed = timestamp - self._calib_start_ts
            self._calib_values.append(ear)
            if elapsed >= self.calib_sec and len(self._calib_values) > 0:
                self.baseline = float(np.median(self._calib_values))
            return {
                "calibrating": True,
                "baseline": self.baseline,
                "relative_ear": None,
                "eye_closed": False,
            }

        # 판정 구간
        rel = ear / self.baseline if self.baseline > 1e-6 else 0.0
        return {
            "calibrating": False,
            "baseline": self.baseline,
            "relative_ear": rel,
            "eye_closed": rel < self.close_ratio,
        }

    def reset(self):
        self._calib_values = []
        self._calib_start_ts = None
        self.baseline = None


# --- 자체 테스트 ---
if __name__ == "__main__":
    # 가짜 눈 좌표로 EAR 계산 검증
    # 뜬 눈: 세로로 넓게
    open_eye = np.array([
        [0, 5],   # p1 바깥끝
        [2, 0],   # p2 위
        [4, 0],   # p3 위
        [6, 5],   # p4 안끝
        [4, 10],  # p5 아래
        [2, 10],  # p6 아래
    ], dtype=np.float32)
    # 감은 눈: 세로로 납작하게
    closed_eye = np.array([
        [0, 5],
        [2, 4],
        [4, 4],
        [6, 5],
        [4, 6],
        [2, 6],
    ], dtype=np.float32)

    ear_open = _eye_aspect_ratio(open_eye)
    ear_closed = _eye_aspect_ratio(closed_eye)
    print(f"뜬 눈 EAR:  {ear_open:.3f}")
    print(f"감은 눈 EAR: {ear_closed:.3f}")
    assert ear_open > ear_closed, "뜬 눈이 감은 눈보다 EAR 커야 함"

    # 68점 배열 만들어서 compute_ear 테스트
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[LEFT_EYE] = open_eye
    lm[RIGHT_EYE] = open_eye
    print(f"양쪽 뜬 눈 평균 EAR: {compute_ear(lm):.3f}")

    # 캘리브레이션 + 판정 시나리오
    judge = EyeStateJudge(calib_sec=3.0, close_ratio=0.6)
    t = 0.0
    dt = 1/30
    # 3초 눈 뜬 상태로 캘리브레이션 (EAR ~0.83)
    r = None
    while t < 3.5:
        r = judge.update(t, ear_open)
        t += dt
    print(f"\n캘리브레이션 완료. baseline={judge.baseline:.3f}")
    assert judge.is_calibrated()

    # 뜬 눈 판정
    r = judge.update(t, ear_open)
    print(f"뜬 눈:  relative={r['relative_ear']:.2f}, closed={r['eye_closed']}")
    assert r['eye_closed'] == False
    # 감은 눈 판정
    r = judge.update(t, ear_closed)
    print(f"감은 눈: relative={r['relative_ear']:.2f}, closed={r['eye_closed']}")
    assert r['eye_closed'] == True

    print("\nEAR 모듈 테스트 통과")
