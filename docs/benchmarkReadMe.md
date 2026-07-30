# benchmark 디렉터리

## 구성

| 파일 | 역할 |
|---|---|
| `performance_logger.py` | 프레임별 시간과 실행 요약 CSV 작성 |
| `evaluate.py` | 정답 구간과 검출 구간의 이벤트/프레임 점수 계산 |
| `compute_metrics.py` | FPS, 지연, Recall, 오탐/분, 경고 지연, 자원 지표 보고 |
| `results/` | 로컬 실험 산출물; 기본적으로 `.gitignore` 대상 |

## 지표 정의

- Event Recall: 정답 이벤트 중 검출 구간과 한 번이라도 겹친 이벤트의 비율
- Frame Precision: 검출한 눈 감김 프레임 중 실제 눈 감김 프레임 비율
- Frame Recall: 실제 눈 감김 프레임 중 검출한 프레임 비율
- False Alarms/min: 정답 구간과 겹치지 않는 검출 이벤트를 영상 분으로 나눈 값
- 위험 경고 지연: 정답 눈 감김 시작부터 `DANGER` 발생까지의 시간
- Model FPS: `1000 / (YuNet mean + PFLD mean)`
- E2E FPS: 모델뿐 아니라 해당 실행이 기록한 프레임 처리 전체 속도

## CSV 관계

```text
*_frames.csv  -- 프레임별 상태/시간
*_summary.csv -- 한 실행의 FPS/지연 요약
labels.csv    -- 수동 정답 이벤트 구간
score_*.csv   -- evaluate.py 결과
metrics_*.csv -- compute_metrics.py 종합 보고
```

## 공정한 백엔드 비교 조건

`benchmark_model_backends.py`는 같은 영상, 30 warm-up 프레임, YuNet 640×640
letterbox, PFLD 112×112를 사용합니다. OpenCV CUDA만 원본 해상도 입력을 쓰면
TensorRT와 전처리 조건이 달라지므로 순수 백엔드 비교가 아닙니다. FSM과 overlay를
확인하는 시스템 시험과 모델 속도 시험도 서로 분리해 해석합니다.

## 현재 대표 결과

대표 표와 그래프는 [ProjectReadMe.md](ProjectReadMe.md)에 있습니다. 전력 원본은
용량과 장치 공유를 위해 `/srv/samba/`에 저장되며 Git에는 포함되지 않습니다.
발표 결과를 고정하려면 사용한 summary/score 파일을 별도 릴리스 산출물로 보존해야
합니다.
