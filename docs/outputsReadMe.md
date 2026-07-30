# outputs 및 logs 디렉터리

## outputs

| 경로 | 내용 |
|---|---|
| `outputs/video_inference/` | 저장 영상 추론의 overlay 영상 |
| `outputs/videos/` | 실시간 카메라 및 일반 추론 녹화 |
| `outputs/ui/` | Qt dashboard 캡처 |
| `outputs/plots/` | 로컬 분석 그래프 |
| `outputs/tables/` | 로컬 분석 표 |

## logs

`logs/`에는 PERCLOS와 런타임 로그, benchmark/tegrastats 로그를 둘 수 있습니다.
성능 CSV의 공식 위치는 `benchmark/results/`, 전력 실험 묶음의 공식 위치는
`/srv/samba/pure_backend_<backend>_runN/`입니다.

## Git 정책

영상, 로그, 출력물은 크기가 크고 장치마다 달라 `.gitignore` 대상입니다.
`outputs/.gitkeep`와 `logs/.gitkeep`만 디렉터리 유지를 위해 추적합니다. 발표에
사용하는 그래프는 재현과 리뷰가 가능하도록 `docs/assets/`에 작은 SVG로 보존합니다.

공유할 때는 다음을 한 세트로 묶습니다.

1. 입력 영상 식별 정보와 실행 명령
2. `*_frames.csv`와 `*_summary.csv`
3. 라벨 및 `score_*.csv`, `metrics_*.csv`
4. 전력 실험이면 tegrastats 원본/요약과 run log
5. 사용 commit, JetPack/CUDA/TensorRT/OpenCV 버전
