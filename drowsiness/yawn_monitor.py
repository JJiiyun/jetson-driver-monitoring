#!/usr/bin/env python3
"""
하품 검출 (MAR 기반).

눈 감김이 EAR(Eye Aspect Ratio)을 쓰듯, 하품은 MAR(Mouth Aspect Ratio,
입 종횡비)로 감지한다. 입이 크게 벌어지면 세로/가로 비율이 커진다.

EyeClosureMonitor와 같은 구조:
  - 캘리브레이션(선택): 평소 입 MAR로 개인 기준 잡기
  - 히스테리시스: 벌어짐/닫힘 판정이 떨리지 않게
  - 지속시간: 하품은 보통 1초 이상 지속되므로 짧은 입벌림(말하기)과 구분

68점 랜드마크 기준 입 좌표 (dlib 표준):
  바깥 입술: 48~59
  안쪽 입술: 60~67
  MAR 계산엔 세로 3쌍 + 가로 1쌍을 쓴다.
"""

from dataclasses import dataclass
import numpy as np


# 68점 모델의 입 랜드마크 인덱스 (dlib 표준)
# 세로 거리용 점쌍 (위입술, 아래입술)
MOUTH_VERTICAL_PAIRS = [(51, 57), (62, 66), (63, 65)]
# 가로 거리용 점쌍 (입 좌우 끝)
MOUTH_HORIZONTAL_PAIR = (48, 54)


@dataclass
class YawnState:
    mar: float            # 현재 MAR 값
    is_yawning: bool      # 하품 중인가
    open_seconds: float   # 연속 입벌림 시간
    status: str           # "NORMAL" 또는 "YAWNING"


def compute_mar(landmarks, vertical_pairs=None, horizontal_pair=None):
    """
    MAR = 평균(세로 거리) / 가로 거리

    landmarks: (N, 2) 배열, 68점이면 (68, 2)
    벌어질수록 MAR이 커진다.
    """
    if vertical_pairs is None:
        vertical_pairs = MOUTH_VERTICAL_PAIRS
    if horizontal_pair is None:
        horizontal_pair = MOUTH_HORIZONTAL_PAIR

    landmarks = np.asarray(landmarks, dtype=np.float32)

    # 세로 거리들의 평균
    v_dists = []
    for (a, b) in vertical_pairs:
        v_dists.append(np.linalg.norm(landmarks[a] - landmarks[b]))
    v_mean = float(np.mean(v_dists))

    # 가로 거리
    (h1, h2) = horizontal_pair
    h_dist = float(np.linalg.norm(landmarks[h1] - landmarks[h2]))

    if h_dist < 1e-6:
        return 0.0
    return v_mean / h_dist


class YawnMonitor:
    """
    MAR 시계열을 받아 하품을 판정한다.
    EyeClosureMonitor와 같은 사용법(update per frame).
    """

    def __init__(
        self,
        open_ratio: float = 0.18,     # MAR이 이 값 이상이면 "벌어짐"
        close_ratio: float = 0.14,    # 이 값 아래로 내려가야 "닫힘" (히스테리시스)
        yawn_seconds: float = 0.3,    # 이 시간 이상 벌어져 있으면 하품
        use_calibration: bool = False,
    ):
        if not close_ratio < open_ratio:
            raise ValueError("close_ratio must be < open_ratio for hysteresis.")
        if yawn_seconds <= 0:
            raise ValueError("yawn_seconds must be positive.")

        self.open_ratio = float(open_ratio)
        self.close_ratio = float(close_ratio)
        self.yawn_seconds = float(yawn_seconds)
        self.use_calibration = bool(use_calibration)

        self._is_open = False              # 현재 입이 벌어진 상태인가 (히스테리시스)
        self._open_started_at = None       # 벌어지기 시작한 시각
        self._baseline_mar = None          # 캘리브레이션 기준 (평소 MAR)

    def calibrate(self, mar_samples):
        """평소(닫힌 입) MAR 값들로 기준을 잡는다. (선택)"""
        if len(mar_samples) > 0:
            self._baseline_mar = float(np.median(mar_samples))

    def update(self, mar: float, timestamp: float) -> YawnState:
        """
        한 프레임의 MAR로 상태를 갱신한다.
        timestamp: 초 단위 (frame_index/fps 권장, 벽시계 아님)
        """
        # 캘리브레이션 보정 (평소 대비 상대값)
        effective_mar = mar
        if self.use_calibration and self._baseline_mar:
            # 평소보다 얼마나 더 벌어졌나 (비율)
            effective_mar = mar / self._baseline_mar if self._baseline_mar > 1e-6 else mar

        # 히스테리시스로 벌어짐/닫힘 판정
        if self._is_open:
            # 벌어진 상태 -> close_ratio 아래로 내려가야 닫힘
            if effective_mar < self.close_ratio:
                self._is_open = False
                self._open_started_at = None
        else:
            # 닫힌 상태 -> open_ratio 이상 올라가야 벌어짐
            if effective_mar >= self.open_ratio:
                self._is_open = True
                self._open_started_at = timestamp

        # 연속 벌어짐 시간 계산
        if self._is_open and self._open_started_at is not None:
            open_seconds = max(0.0, timestamp - self._open_started_at)
        else:
            open_seconds = 0.0

        # 하품 판정: 일정 시간 이상 연속으로 벌어져 있으면
        is_yawning = open_seconds >= self.yawn_seconds
        status = "YAWNING" if is_yawning else "NORMAL"

        return YawnState(
            mar=mar,
            is_yawning=is_yawning,
            open_seconds=open_seconds,
            status=status,
        )

    def reset(self):
        self._is_open = False
        self._open_started_at = None


if __name__ == "__main__":
    # 간단 자체 테스트
    monitor = YawnMonitor(open_ratio=0.18, close_ratio=0.14, yawn_seconds=0.3)
    fps = 30.0
    print("프레임별 하품 판정 테스트:")
    # 0~0.5초 다문 입, 0.5~2.5초 하품(벌어짐), 이후 다시 닫힘
    for i in range(90):
        t = i / fps
        if 0.5 <= t <= 2.5:
            mar = 0.75   # 벌어짐
        else:
            mar = 0.3    # 다뭄
        st = monitor.update(mar, t)
        if i % 15 == 0 or st.is_yawning != (i > 0 and prev):
            print(f"  t={t:.2f}s MAR={mar:.2f} 벌어짐={monitor._is_open} "
                  f"연속={st.open_seconds:.2f}s 하품={st.is_yawning}")
        prev = st.is_yawning
