# MSG-339 AI 블러 처리량 개선 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task.

**목표:** 전용 t3.small에서 블러 파이프라인의 단일 처리 시간을 20% 이상 줄이고 동시 6건 처리량을 1.7배 이상 높인다.

**구조:** uvicorn 프로세스 하나에서 인메모리 FIFO 큐[^1]를 공유한다. 워커 스레드는 최대 2개만 띄우고, 각 작업은 자기 모델을 로드한다. 한 작업 안에서는 프레임을 최대 4장만 모아 배치 추론[^2]한 뒤 입력 순서대로 쓴다.

**기술:** Python, FastAPI, `queue.Queue`, `threading`, OpenCV, Ultralytics YOLO, Bash, Docker, AWS EC2

## 공통 제약

- 요구사항 정본은 `docs/prd/MSG-339-prd.md`, 구현 정본은 `docs/MSG-339.md`다.
- `AI_WORKERS`는 `1` 또는 `2`, `AI_BATCH_SIZE`는 `1`, `2`, `4`만 허용한다. 기본값은 모두 `1`이다.
- HTTP 경로, 응답 필드, 상태 전이는 바꾸지 않는다.
- 모델을 워커 사이에서 공유하지 않고 새 의존성도 추가하지 않는다.
- 에이전트는 커밋하지 않는다. 각 작업의 검증과 리뷰가 끝나면 성민의 커밋 지점에서 멈춘다.
- 성능 판정은 전용 EC2 실측만 인정한다. 로컬 smoke는 기능 회귀만 확인한다.

---

## 작업 1: 프레임 배치 추론

**파일:**

- 수정: `bench.py`

### 1. 실패하는 smoke 검증 추가

`smoke()`의 합성 영상을 30fps 기준 4로 나누어떨어지지 않는 길이로 바꾼다. 리포트에는 적용된 배치 크기와 모델 호출 배치 수를 남기고 다음 성질을 검사한다.

```python
assert report["batch_size"] == AI_BATCH_SIZE
assert report["inference_batches"] == (report["frames"] + AI_BATCH_SIZE - 1) // AI_BATCH_SIZE
```

먼저 아래 명령을 실행해 새 필드가 없어 실패하는지 확인한다.

```bash
AI_BATCH_SIZE=4 ./.venv/bin/python bench.py --smoke
```

### 2. 허용값 검증

모듈을 불러올 때 환경변수를 읽고 허용값이 아니면 `ValueError`로 시작을 막는다.

```python
AI_BATCH_SIZE = int(os.environ.get("AI_BATCH_SIZE", "1"))
if AI_BATCH_SIZE not in (1, 2, 4):
	raise ValueError("AI_BATCH_SIZE는 1, 2, 4만 허용")
```

### 3. 제한된 버퍼로 추론

`run()`은 최대 `AI_BATCH_SIZE`개 프레임만 `pending`에 보관한다. 각 묶음에서 추론할 프레임 목록을 얼굴 모델과 번호판 모델에 같은 순서로 전달하고, 결과 목록을 같은 인덱스로 소비한다. `AI_INFER_STRIDE`가 1보다 큰 기존 실험 경로에서는 묶음 안의 추론 대상만 모델에 넘기고 나머지는 직전 확장 상자를 재사용한다.

```python
pending = []
while len(pending) < AI_BATCH_SIZE:
	ok, frame = cap.read()
	if not ok:
		break
	pending.append(frame)
```

EOF에서 `pending`이 비지 않았으면 같은 처리 경로를 한 번 더 탄다. 영상 전체 프레임을 리스트로 만들지 않는다.

### 4. 통과 확인

```bash
AI_BATCH_SIZE=4 ./.venv/bin/python bench.py --smoke
AI_BATCH_SIZE=3 ./.venv/bin/python -c 'import bench'
```

첫 명령은 통과해야 한다. 두 번째 명령은 허용하지 않은 값이라 실패해야 한다. 기존 기본값도 확인한다.

```bash
./.venv/bin/python bench.py --smoke
```

### 5. 리뷰와 커밋 정지점

검토 담당은 마지막 불완전 배치, 프레임 수, `AI_INFER_STRIDE` 회귀, 모델 결과 순서를 확인한다. 지적을 반영하고 위 명령을 다시 통과한 뒤 성민의 첫 커밋을 기다린다.

---

## 작업 2: 워커 두 개로 작업 동시 처리

**파일:**

- 수정: `server.py`

### 1. 실패하는 smoke 검증 추가

`smoke()`는 정상 영상 두 건을 연달아 제출한 뒤 둘 다 `DONE`인지 확인한다. 손상된 입력 한 건은 `FAILED`여야 하고 앞선 두 작업의 결과에 영향을 주면 안 된다. 시작된 스레드 수가 설정값과 같은지도 검사한다.

```python
assert len(worker_threads) == AI_WORKERS
assert all(thread.is_alive() for thread in worker_threads)
```

구현 전에 다음 명령을 실행해 새 설정이나 스레드 목록이 없어 실패하는지 확인한다.

```bash
AI_WORKERS=2 AI_BATCH_SIZE=4 ./.venv/bin/python server.py --smoke
```

### 2. 워커 수 검증과 시작

`AI_WORKERS`를 읽어 `1`과 `2`만 허용한다. 기존 `worker()`를 바꾸지 않고 설정한 수만큼 데몬 스레드를 만든다.

```python
AI_WORKERS = int(os.environ.get("AI_WORKERS", "1"))
if AI_WORKERS not in (1, 2):
	raise ValueError("AI_WORKERS는 1, 2만 허용")

worker_threads = [threading.Thread(target=worker, daemon=True) for _ in range(AI_WORKERS)]
for thread in worker_threads:
	thread.start()
```

전역 `jobs`와 `job_queue`는 그대로 공유한다. 프로세스 풀, 잠금, 별도 저장소는 만들지 않는다.

### 3. 통과 확인

```bash
AI_WORKERS=2 AI_BATCH_SIZE=4 ./.venv/bin/python server.py --smoke
AI_WORKERS=3 ./.venv/bin/python -c 'import server'
```

첫 명령은 정상 두 건과 손상 입력의 독립성을 확인한다. 두 번째 명령은 시작 오류가 나야 한다.

### 4. 리뷰와 커밋 정지점

검토 담당은 같은 job을 두 워커가 처리할 수 없는지, 한 작업의 예외가 다른 워커를 끝내지 않는지, API 계약이 그대로인지 확인한다. 통과 후 성민의 두 번째 커밋을 기다린다.

---

## 작업 3: EC2 배포와 실측 스크립트

**파일:**

- 수정: `scripts/ec2-bench.sh`
- 생성: `scripts/ec2-ai-load-test.sh`
- 수정: `scripts/ec2-deploy.sh`
- 유지: `scripts/ec2-load-test.sh`

### 1. 배치 후보 벤치마크

`scripts/ec2-bench.sh`는 같은 `crowd`, `plates`, 한국 샘플 프레임에 배치 1, 2, 4를 적용한다. 후보별 3회 중앙값과 피크 메모리를 저장한다. `face_experiment.py`의 `iou()`를 재사용해 얼굴과 번호판 상대 recall이 각각 1.000인지 판정한다.[^3]

### 2. AI API 전용 동시 부하

새 `scripts/ec2-ai-load-test.sh`는 `N=6`을 기본값으로 한다. 상태는 2초, 메모리, 스왑, `vmstat`의 `si`와 `so`, load1은 10초마다 기록한다. 모든 작업의 `DONE`, 결과 크기, `ffprobe`, Docker `RestartCount`, `OOMKilled`를 한 요약 파일에 남긴다.[^4]

### 3. 배포 설정과 실영상 두 건

`scripts/ec2-deploy.sh`는 `$HOME/fillmap-ai.env`의 `AI_WORKERS`, `AI_BATCH_SIZE`를 읽고 없으면 `1`, `1`을 쓴다. Docker `-e`로 전달한 뒤 `plates` 실영상 두 건을 먼저 제출하고 둘 다 `DONE`인지 확인한다.

### 4. 정적 검증

```bash
bash -n scripts/ec2-bench.sh
bash -n scripts/ec2-ai-load-test.sh
bash -n scripts/ec2-deploy.sh
git diff --check
```

검토가 끝나면 성민의 세 번째 커밋을 기다린다.

---

## 작업 4: 전용 EC2 실측과 채택값 확정

**파일:**

- 생성: `results/MSG-339-report.md`
- 생성: `results/MSG-339-*.json`
- 수정: `README.md`
- 수정: `CLAUDE.md`

### 1. 전용 인스턴스 준비

서울 리전의 기존 BE와 같은 VPC 및 가용 영역에 t3.small, gp3 20GB, 2GB 영구 스왑을 구성한다. 8000 포트는 BE 보안 그룹만 허용한다. GitHub Actions의 `DEV_EC2_HOST`와 BE의 `AI_BASE_URL`을 새 주소로 바꾼다.

### 2. 순서대로 실측

1. 워커 1, 배치 1의 단일 3회와 동시 6건 기준선을 잰다.
2. 워커 1에서 배치 2와 4를 각각 3회 잰다.
3. 단일 시간 20% 단축과 상대 recall 1.000을 만족한 가장 작은 배치를 고른다.
4. 워커 2와 채택 배치로 동시 6건을 실행한다.
5. 처리량 1.7배, 개별 처리 시간 증가 30% 이하, 유실과 OOM 0건, 연속 스왑 3회 미만을 모두 확인한다.

한 항목이라도 실패하면 `AI_WORKERS=1`, `AI_BATCH_SIZE=1`로 되돌리고 실패 결과도 보고서에 남긴다.

### 3. 비용과 문서 마감

같은 측정 구간의 CPU 크레딧과 인스턴스, gp3, EIP 비용을 기록한다. 채택값과 롤백 명령만 `README.md`, `CLAUDE.md`에 반영한다. 전체 로컬 smoke, 셸 문법, `git diff --check`를 다시 실행한 뒤 마지막 커밋 지점에서 멈춘다.

[^1]: FIFO는 먼저 들어온 작업을 먼저 꺼내는 큐다. 워커가 둘이면 큐에서 꺼내는 순서는 유지되지만 완료 순서는 달라질 수 있다.
[^2]: 배치 추론은 여러 프레임을 모델 호출 한 번에 넘기는 방식이다. 여기서는 메모리를 제한하려고 한 번에 최대 4장만 보관한다.
[^3]: IoU는 두 검출 상자의 교집합 면적을 합집합 면적으로 나눈 값이다. 0.5 이상이면 같은 대상을 찾았다고 보고 배치 1 대비 누락 여부를 계산한다.
[^4]: OOM은 메모리가 부족해 운영체제나 런타임이 프로세스를 종료하는 상태다. Docker의 `OOMKilled`와 재시작 횟수로 확인한다.
