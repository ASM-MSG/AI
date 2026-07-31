# MSG-207 — AI 추론 최적화 실측 (OpenVINO · 프레임 스킵)

2026-07-23, dev EC2(t3.small, x86_64 2vCPU) `fillmap-ai` 컨테이너 내 측정.
목표는 건당 3분 → 1분. **결론: 두 방법 모두 미채택** — 속도는 나오지만 안전하지 않다.

## 1. OpenVINO 변환 — 미채택 (+5%뿐)

crowd 29초 1080p(696프레임), 전 프레임 추론:

| 백엔드 | wall | 추론 ms/frame | 검출 (face/plate) |
|---|---|---|---|
| torch (기준선) | 180.5s | 200.9 | 9,943 / 37 |
| openvino | 171.3s | 191.6 | 9,983 / 50 |

- 검출 동등성은 확인(0.4% 차이)되지만 속도 이득이 **~5%**. torch가 이미 oneDNN으로
  2vCPU를 포화시키고 있어 변환 여지가 없다. "x86에서 2~3배"는 이 환경에선 신화.
- 의존성(+이미지 ~100MB)·export 단계 추가 대비 무가치 → **requirements에서 제외**.
  `bench.py`의 `AI_BACKEND=openvino` 노브는 실험용으로 남긴다 (별도 `pip install openvino`).

## 2. 프레임 스킵 — 미채택 (도로 씬에서 커버리지 붕괴)

전 프레임 검출을 pseudo-GT로, 스킵 프레임을 직전 앵커 박스 확장본으로 가리는 정책을
시뮬레이션 (`stride_experiment.py`, 원자료 `results/MSG-207-stride.json`).
수치는 "GT 박스 면적 80% 이상 가려진 비율"(covered_rate).

**crowd (고정 카메라, 사람 빽빽)** — 얼굴은 그럭저럭 버틴다:

| 설정 | face | plate |
|---|---|---|
| stride 2 + pad 0.2 | **96.1%** | 72%* |
| stride 3 + pad 0.2 | 92.9% | 57.6%* |

**plates (도로 정면, 차량 접근)** — 완전히 무너진다:

| 설정 | face | plate |
|---|---|---|
| stride 2 + pad 0.2 | 29.6% | **62.3%** |
| stride 3 + pad 0.2 | 14.7% | 46.1% |

\* crowd의 plate 표본은 배경 차량 깜빡임 위주(50건)라 참고치. plates 표가 정본.

- 원인: 접근하는 차량의 번호판은 프레임당 수십 px씩 이동·확대 — 박스 재사용이 못 따라간다.
  도로 씬의 작은 얼굴(행인·운전자)도 마찬가지.
- 속도는 예측대로: openvino+stride3에서 crowd 90.0s(기준선의 절반, 추론 65ms/frame),
  plates 73.9s. **하지만 번호판 블러가 가장 중요한 씬이 바로 도로 씬이다** —
  "미탐지는 프라이버시 사고" 원칙상 채택 불가. `AI_INFER_STRIDE` 기본값 1 유지.

## 3. 판단과 다음 레버

**소프트웨어 레버 둘 다 기각. 건당 ~3분(웜)은 t3.small 2vCPU에서 YOLOv11n×2모델의
정직한 비용이다.** 속도가 정말 필요해지면 순서는:

1. **모션 게이트 적응 스킵** — 앵커 대비 프레임 차분이 작을 때만 스킵(정적 씬 한정).
   도로 씬은 자동으로 전 프레임 추론으로 폴백. 별도 커버리지 검증 실험 필요.
2. **배치 추론** — k프레임 묶어 predict. 호출당 오버헤드 절감분 실측 필요 (메모리 주의).
3. **하드웨어** — c6i.large(2vCPU 고정 클럭)~xlarge(4vCPU, ~2배), GPU 스팟(g4dn, 수십 배).
   비용 시뮬레이션과 함께 별도 티켓.

재현: 컨테이너에서 `AI_BACKEND=openvino AI_INFER_STRIDE=3 python bench.py crowd.mp4`,
시뮬레이션은 `AI_BACKEND=openvino python stride_experiment.py crowd.mp4 plates.mp4`.
