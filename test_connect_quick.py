from drowsiness.calibration import EyeClosureMonitor
from drowsiness.perclos_monitor import PerclosMonitor

eye = EyeClosureMonitor()
perclos = PerclosMonitor(window_seconds=30.0)

es = ps = None
for i in range(200):
    t = i * 0.1
    ear = 0.30 if t < 5 else 0.15    # 5초 뒤부터 눈 감음
    es = eye.update(ear, timestamp=t)
    ps = perclos.update(is_closed=es.is_closed,
                        valid_face=es.valid_face,
                        timestamp=t)

print("눈 상태:", es.status)
print("PERCLOS:", round(ps.perclos, 3))
print("경고?:", ps.is_warning)
