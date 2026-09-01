# FillMap AI

FillMap의 AI Highlight-Blur 서버. 업로드된 영상에서 얼굴·번호판을 자동 블러 처리하고,
하이라이트 구간을 추천한다.

백엔드(Spring Boot)는 [ASM-MSG/BE](https://github.com/ASM-MSG/BE)에 있고, 이 레포와는
HTTP로만 통신하는 별도 프로세스다.

## 라이선스가 왜 AGPL-3.0인가

이 레포는 Ultralytics YOLOv11을 쓰고, Ultralytics는 AGPL-3.0이다. AGPL 13조는 배포하지
않고 네트워크 서비스로만 제공해도 소스 공개 의무를 발생시킨다(상업 여부 무관).

BE 레포와 프로세스·저장소를 분리한 이유가 이것이다. 전염 경계를 여기서 끊어
`ASM-MSG/BE`는 MIT를 유지한다. 결정 근거는 BE 레포의 `docs/MSG-144.md` 참고.

## 모델

| 대상 | 모델 |
|---|---|
| 얼굴 | [`AdamCodd/YOLOv11n-face-detection`](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) (WIDER FACE) |
| 번호판 | [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) |
| 하이라이트 | PySceneDetect (룰 기반) |

가중치는 첫 실행 시 Hugging Face에서 자동 다운로드된다.

## 서버 (MSG-161, MSG-339)

FastAPI 상시 서버 (MSG-143 ADR). BE와 분리한 AI 전용 dev EC2에 Docker 컨테이너로 배포한다.
기본값은 워커[^1] 1개와 배치[^2] 1개다. 전용 t3.small 실측에서 배치 2와 4가 각각 4.34%,
3.94% 느려 기각했으며, 실측 원자료와 판정은 `results/MSG-339-report.md`에 남겼다.

```bash
./.venv/bin/python server.py --smoke     # 합성 영상으로 API 왕복 검증
./.venv/bin/uvicorn server:app --port 8000

docker build -t fillmap-ai .
docker run -p 8000:8000 -e AI_WORKERS=1 -e AI_BATCH_SIZE=1 \
  -v hf-cache:/root/.cache/huggingface fillmap-ai
```

`AI_WORKERS`는 `1` 또는 `2`, `AI_BATCH_SIZE`는 `1`, `2`, `4`만 받는다. 현재 배포값과
롤백값은 모두 `1`이다. t3 계열은 CPU 크레딧[^3]을 다 쓰고도 계속 버스트하면 추가 비용이
생길 수 있으므로 처리량 실측과 함께 확인한다.

### 배포 (MSG-282)

**`main`에 머지하면 자동 배포된다.** GitHub Actions(`.github/workflows/cd-dev.yml`)가
`scripts/ec2-deploy.sh`를 AI 전용 dev EC2에서 실행한다. pull → 이미지 빌드 → 컨테이너 교체 →
`/health` 대기 → 실영상 E2E. E2E까지 통과해야 초록불이라, 뜨자마자 죽은 경우를 잡는다.

수동으로 돌릴 때(Actions 없이, 또는 재배포):

```bash
gh workflow run "CD (dev)" --repo ASM-MSG/AI      # 워크플로 수동 트리거
# 또는 EC2에서 직접
scp scripts/ec2-deploy.sh <user>@<host>:~/ && ssh <user>@<host> 'bash ec2-deploy.sh'
```

배포 대상은 secrets(`DEV_EC2_HOST`/`DEV_EC2_USER`/`DEV_EC2_SSH_KEY`)로 주입한다.
`DEV_EC2_HOST`는 AI 전용 인스턴스의 EIP를 가리킨다.

### API (BE ↔ AI 계약)

처리는 비동기다. 1080p 30초 기준 3~4분 걸린다(실측). BE는 업로드 후
상태를 폴링해 `processing_status`를 갱신한다.

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/jobs` | multipart `file`로 영상 업로드. 즉시 `202 {job_id, status}` |
| `GET` | `/jobs/{id}` | `{job_id, status, highlights, error, precheck}` (폴링용) |
| `GET` | `/jobs/{id}/video` | 블러 처리본 mp4 (h264, 원본 오디오 유지). 완료 전 409 · **프리체크 탈락 409** |
| `POST` | `/highlights` | multipart `file`. **블러 없이 하이라이트만 동기 계산** → `200 {"highlights": [[시작초, 끝초], …]}`. 열 수 없는 입력은 `422` (MSG-353, 업로드 확정 전 선분석용) |
| `GET` | `/health` | 컨테이너 헬스체크 |
| `POST` | `/route/parse` | JSON `{text, viewport}`. 자연어를 `{region, period, interests, preferred_order, related}`로 동기 해석. 형태 위반 출력은 서버가 걸러 `502` (MSG-458) |
| `POST` | `/route/explain` | JSON `{points, text?}`. 지점별 추천 이유 한 줄을 같은 개수·순서로 동기 반환하고, `text`(사용자 문장, 선택)가 있으면 동선 전체 종합 이유 `summary`(1~240자)를 함께 싣는다 (MSG-540). 플래그 off는 두 경로 다 `503` (MSG-458) |

`status` 전이: `QUEUED → PROCESSING → DONE | FAILED`. BE 매핑 예:
`PROCESSING`이면 `processing_status = BLURRING`. `highlights`는 완료 시
`[[시작초, 끝초], …]` 최대 3구간.

하이라이트 구간 품질(MSG-353, `/jobs`·`/highlights` 공통): 각 구간 5초 이상,
구간 시작점끼리 5초 이상 간격, 배열 순서 = 추천 우선순위(첫 요소가 최우선).
조건을 지키며 3구간을 못 채우면 개수를 줄인다. **5초 미만 영상은 빈 배열**.

`precheck`(MSG-284)는 추론 전 무의미 영상(암흑·렌즈 가림) 판정 결과다.
`{passed: bool, reason: string|null}`, 판정 전에는 `null`.
탈락 잡은 **`status=DONE` · `highlights=[]` · `precheck.passed=false`**이고 블러본이 없어
`/video`가 **409**다(블러 안 한 원본을 대신 내보내지 않는다). BE가 `precheck`를 무시해도
정상 영상 경로의 동작은 그대로다. 판정 규칙: `results/MSG-284-report.md`.

파이프라인: **1080p 30fps 다운스케일**(ADR 전제, 초과분만 축소하고 업스케일하지 않음 —
이미 기준 이내면 재인코딩 없이 원본을 그대로 분석) → **프리체크**(탈락이면 여기서 끝)
→ 얼굴·번호판 블러 → 하이라이트. 재생본은 블러 프레임을 rawvideo 파이프로 ffmpeg에 흘려
**단일 인코딩 패스**(h264 + 원본 오디오)로 굽는다 — 별도 병합 패스와 mp4v 중간 파일이
없다(MSG-367, 세대 손실 제거). 현재 잡은 큐에서 순차 처리한다(`AI_WORKERS=1`).

경로 추천 언어 처리(MSG-458)는 블러 잡과 독립인 **동기** 경로다 — 워커 큐를 거치지 않고
잡 상태도 만들지 않는다. `/route/parse`는 `{text(1~500자), viewport(WGS84 사각형)}`를 받아
`{region, period{start, end}, interests[], preferred_order[], related}`로 해석하고(전 필드 빈
결과도 유효 — 못 읽었다는 뜻. `related`는 별개 축으로, 장소 방문 동선 요청이 확실히 아닐 때만
false — MSG-533), `/route/explain`은 `{points[{name, kind, facts[]}], text?}`(지점 1~20개,
`facts`는 지점당 1~5개, `text`는 사용자 문장 원문으로 선택 1~500자 — MSG-540)를 받아
지점과 같은 개수·순서의 `{reasons[]}`(각 개행 없는 1~120자)를 돌려주고, `text`가 있으면
동선 전체의 종합 이유 `summary`(개행 없는 1~240자)를 함께 싣는다(없으면 키 자체가 없다). 장소 선정은 BE 몫이고
이 서버는 문장↔구조 번역만 한다. 모델(OpenAI `ROUTE_AI_MODEL`, 기본 `gpt-4o-mini`) 출력이
계약 형태를 벗어나면(비 JSON·미정의 필드·타입·길이·개수 위반) 200으로 흘리지 않고 `502`로
끝낸다. 실패는 네 가지뿐이다 — 요청 오류 `422` · 모델 실패/형태 위반 `502` ·
`ROUTE_AI_ENABLED` 꺼짐(기본) `503` · 모델 타임아웃(`ROUTE_AI_TIMEOUT_SEC`, 기본 30초) `504`.
BE는 502·503·504를 전부 "해석 실패"로 받는다. 두 계약 어디에도 사용자 식별 정보 필드는 없다.
켜려면 `ROUTE_AI_ENABLED=1`과 `ROUTE_AI_API_KEY`가 둘 다 필요하다(키 없이 켜면 기동 실패).

## 벤치마크 (MSG-142)

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

./.venv/bin/python bench.py --smoke              # 합성 영상으로 파이프라인 검증
./.venv/bin/python bench.py samples/normal.mp4 --device cpu
./.venv/bin/python bench.py samples/normal.mp4 --device cuda   # AWS GPU
```

단계별(디코딩 / 얼굴 추론 / 번호판 추론 / 마스킹 / 재인코딩) 시간을 따로 재서
병목이 AI인지 인코딩인지 가른다.

`samples/`는 gitignore 대상이다. 실제 얼굴·번호판이 담긴 영상은 커밋하지 않는다.

[^1]: 워커는 큐에서 작업을 꺼내 블러 파이프라인을 실행하는 단위다. 값이 1이면 여러 요청이 와도 한 건씩 처리한다.
[^2]: 배치 추론은 여러 프레임을 묶어 모델 호출 한 번에 넘기는 방식이다. 이 서버의 CPU 환경에서는 호출 횟수 감소보다 프레임 묶음 처리 비용이 더 컸다.
[^3]: CPU 크레딧은 t3 계열이 기준 CPU 성능을 넘겨 쓸 때 소모하는 값이다. 24시간 평균이 기준을 넘으면 부족분에 요금이 붙을 수 있다.
