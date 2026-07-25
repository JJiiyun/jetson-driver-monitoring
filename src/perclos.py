"""
시간 기반 PERCLOS 계산 모듈.

PERCLOS = 최근 window_sec 동안 눈이 감긴 누적 시간
          ÷ 최근 window_sec 동안 얼굴이 유효하게 검출된 시간

프레임 개수가 아니라 실제 경과 시간(초)으로 누적한다.
FP32와 FP16의 처리 FPS가 달라도 동일 기준으로 비교하기 위함이다.

이 모듈은 얼굴 검출/EAR 계산과 독립적이다.
매 프레임마다 (현재 시각, 얼굴 검출 여부, 눈 감김 여부)만 넘기면
현재 PERCLOS 값과 연속 눈 감김 시간을 돌려준다.
"""

from collections import deque


class PerclosCalculator:
    def __init__(self, window_sec=30.0):
        """
        window_sec: PERCLOS를 계산할 슬라이딩 윈도우 길이(초). 계획서 기준 30초.
        """
        self.window_sec = window_sec
        # 각 원소: (timestamp, dt, face_valid, eye_closed)
        # dt = 직전 프레임과의 시간 간격(초). 이 dt가 시간 누적의 단위가 된다.
        self._events = deque()
        self._last_ts = None
        # 연속 눈 감김 측정용: 눈이 감기기 시작한 시각
        self._closed_start_ts = None

    def update(self, timestamp, face_valid, eye_closed):
        """
        한 프레임 처리.

        timestamp: 현재 프레임 시각(초). time.time() 또는 영상 기준 경과초.
        face_valid: 이번 프레임에서 얼굴이 유효하게 검출됐는지 (bool)
        eye_closed: 이번 프레임에서 눈이 감겼는지 (bool). 얼굴 미검출이면 의미 없음.

        반환: dict
            perclos            : 현재 PERCLOS 값 (0.0 ~ 1.0), 유효 시간 0이면 0.0
            continuous_closed  : 현재 연속으로 눈을 감고 있는 시간(초). 감고 있지 않으면 0.0
            valid_time         : 윈도우 내 얼굴 유효 검출 누적 시간(초)
            closed_time        : 윈도우 내 눈 감김 누적 시간(초)
        """
        # 첫 프레임은 dt를 잴 수 없으므로 0으로 두고 기준 시각만 저장
        if self._last_ts is None:
            dt = 0.0
        else:
            dt = timestamp - self._last_ts
            # 시각이 거꾸로 가거나(영상 되감기 등) 비정상이면 0으로 보정
            if dt < 0:
                dt = 0.0
        self._last_ts = timestamp

        # 눈 감김은 얼굴이 검출됐을 때만 유효하게 센다
        counted_closed = face_valid and eye_closed

        self._events.append((timestamp, dt, face_valid, counted_closed))

        # 윈도우 밖(오래된) 이벤트 제거
        self._evict_old(timestamp)

        # 연속 눈 감김 시간 갱신
        continuous_closed = self._update_continuous_closed(timestamp, counted_closed)

        # 윈도우 내 누적 시간 집계
        valid_time = 0.0
        closed_time = 0.0
        for _ts, ev_dt, ev_face, ev_closed in self._events:
            if ev_face:
                valid_time += ev_dt
            if ev_closed:
                closed_time += ev_dt

        perclos = (closed_time / valid_time) if valid_time > 0 else 0.0

        return {
            "perclos": perclos,
            "continuous_closed": continuous_closed,
            "valid_time": valid_time,
            "closed_time": closed_time,
        }

    def _evict_old(self, now_ts):
        """윈도우(now - window_sec)보다 오래된 이벤트를 앞에서 제거."""
        cutoff = now_ts - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _update_continuous_closed(self, timestamp, counted_closed):
        """
        연속 눈 감김 시간 추적.
        감기 시작 시점을 기억해두고, 계속 감겨 있으면 현재까지의 경과를 반환한다.
        한 번이라도 뜨면 리셋.
        """
        if counted_closed:
            if self._closed_start_ts is None:
                self._closed_start_ts = timestamp
            return timestamp - self._closed_start_ts
        else:
            self._closed_start_ts = None
            return 0.0

    def reset(self):
        """캘리브레이션 재시작 등으로 상태를 완전히 초기화."""
        self._events.clear()
        self._last_ts = None
        self._closed_start_ts = None
