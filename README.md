# FillMap AI

FillMap의 AI Highlight-Blur 서버. 업로드된 영상에서 얼굴·번호판을 자동 블러 처리하고,
하이라이트 구간을 추천한다.

백엔드(Spring Boot)는 [ASM-MSG/BE](https://github.com/ASM-MSG/BE)에 있고, 이 레포와는
HTTP로만 통신하는 별도 프로세스다.

## 라이선스가 왜 AGPL-3.0인가

이 레포는 Ultralytics YOLOv11을 쓰고, Ultralytics는 AGPL-3.0이다. AGPL 13조는 배포하지
않고 네트워크 서비스로만 제공해도 소스 공개 의무를 발생시킨다(상업 여부 무관).

BE 레포와 프로세스·저장소를 분리한 이유가 이것이다 — 전염 경계를 여기서 끊어
`ASM-MSG/BE`는 MIT를 유지한다. 결정 근거는 BE 레포의 `docs/MSG-144.md` 참고.

## 모델

| 대상 | 모델 |
|---|---|
| 얼굴 | [`AdamCodd/YOLOv11n-face-detection`](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) (WIDER FACE) |
| 번호판 | [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) |
| 하이라이트 | PySceneDetect (룰 기반) |

가중치는 첫 실행 시 Hugging Face에서 자동 다운로드된다.

## 서버 (MSG-161)

FastAPI 상시 서버 (MSG-143 ADR). dev EC2에 Docker 컨테이너로 배포한다.

```bash
./.venv/bin/python server.py --smoke     # 합성 영상으로 API 왕복 검증
./.venv/bin/uvicorn server:app --port 8000

docker build -t fillmap-ai .
docker run -p 8000:8000 -v hf-cache:/root/.cache/huggingface fillmap-ai
```

### API (BE ↔ AI 계약)

처리는 비동기다 — 1080p 30초 기준 3~4분 걸린다(실측). BE는 업로드 후
상태를 폴링해 `processing_status`를 갱신한다.

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/jobs` | multipart `file`로 영상 업로드. 즉시 `202 {job_id, status}` |
| `GET` | `/jobs/{id}` | `{job_id, status, highlights, error}` — 폴링용 |
| `GET` | `/jobs/{id}/video` | 블러 처리본 mp4 (h264, 원본 오디오 유지). 완료 전 409 |
| `GET` | `/health` | 컨테이너 헬스체크 |

`status` 전이: `QUEUED → PROCESSING → DONE | FAILED`. BE 매핑 예:
`PROCESSING`이면 `processing_status = BLURRING`. `highlights`는 완료 시
`[[시작초, 끝초], …]` 최대 3구간.

파이프라인: **1080p 30fps 다운스케일**(ADR 전제 — 초과분만 축소, 업스케일 없음)
→ 얼굴·번호판 블러 → 하이라이트 → h264 재인코딩 + 오디오 복원.
잡은 큐로 순차 처리한다(1 vCPU 전제, 시간당 약 17건 상한).

## 벤치마크 (MSG-142)

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python bench.py --smoke              # 합성 영상으로 파이프라인 검증
./.venv/bin/python bench.py samples/normal.mp4 --device cpu
./.venv/bin/python bench.py samples/normal.mp4 --device cuda   # AWS GPU
```

단계별(디코딩 / 얼굴 추론 / 번호판 추론 / 마스킹 / 재인코딩) 시간을 따로 재서
병목이 AI인지 인코딩인지 가른다.

`samples/`는 gitignore 대상이다 — 실제 얼굴·번호판이 담긴 영상은 커밋하지 않는다.
