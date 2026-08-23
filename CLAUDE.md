# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

FillMap의 AI Highlight-Blur 서버. 얼굴·번호판 자동 블러 + 하이라이트 추천.
백엔드(Spring Boot, [ASM-MSG/BE](https://github.com/ASM-MSG/BE))와는 **HTTP로만 통신하는 별도 프로세스**다.

- 스택: Python · ultralytics(YOLOv11n) · PySceneDetect · ffmpeg · (예정) FastAPI
- 실행 환경: AI 전용 dev EC2(t3.small)에 Docker 컨테이너 상시 서버 — BE 레포
  `docs/MSG-143.md` (ADR), `results/MSG-339-report.md` 확정. BE는 사설 IP로 호출한다
- 배포: **`main` 머지 시 자동**(MSG-282, `.github/workflows/cd-dev.yml`). 실체는
  `scripts/ec2-deploy.sh`이고 실영상 E2E까지 통과해야 초록불이다
- 모델 선정 근거: BE 레포 `docs/MSG-144.md`
- 측정 데이터: `results/MSG-142-report.md` — 처리 시간·메모리·발견된 문제 전부 여기

## Read First — 항상 준수 (규칙)

작업 전 반드시 확인:

- `@.claude/rules/coding-principles.md` — 코딩 행동 원칙 (Karpathy 4원칙)
- `@.claude/rules/project-conventions.md` — 파이썬 컨벤션(하드탭) · 검증 관행(`--smoke`·실측)
- `@.claude/rules/subagent-orchestration.md` — 서브에이전트 팀 운영 원칙 (에이전트 팀 스킬 실행 시)

## 라이선스 — 절대 규칙

이 레포는 **AGPL-3.0**이다 (Ultralytics 전염). BE는 MIT를 유지해야 하므로:

- BE 코드와의 통신은 **HTTP만**. 이 레포의 코드를 BE에 복사·import 금지
- 새 의존성 추가 시 라이선스 확인 — AGPL/GPL은 여기 OK, 단 BE로 새어나가면 안 됨
- `samples/`는 gitignore 대상 — 실제 얼굴·번호판이 담긴 영상은 커밋하지 않는다

## 브랜치 · 커밋 (BE 레포와 동일)

git flow 브랜치 타입만 쓴다. 커밋 타입(`feat`·`chore`·`docs`…)을 브랜치 접두어로 쓰지 않는다.

```text
feature/MSG-{번호}-{짧은-설명}   # 일반 작업 전부 — 티켓번호와 설명은 하이픈으로
hotfix/MSG-{번호}-{짧은-설명}
release/{버전}
```

- 작업은 **항상 새 브랜치에서 시작한다.** `main`에 직접 커밋 금지 — 문서 작업도 예외 없다
- 커밋 메시지: `MSG-{번호} {타입}: {요약}` (타입: feat, fix, refactor, docs, test, chore, style)
- 커밋은 성민이 직접 한다. Claude는 커밋 계획(파일 목록 + 메시지)만 제시

## 개발 파이프라인 — PRD 필수

```text
아이디어/티켓 → PRD(docs/prd/*.md) → 스펙(docs/MSG-XXX.md) → 구현
                 ↑ 필수 게이트
```

**PRD 없이 스펙·구현에 착수하지 않는다.** PRD가 없으면 `prd-writer`를 먼저 실행한다 —
`spec-writer`·`spec-driven-dev` 양쪽 진입부에 게이트가 있고, **스펙이 이미 있어도 PRD가 없으면
통과가 아니다**(구 티켓 이어받기가 이 구멍으로 샌다).

**면제 기준 = "이 작업이 제품 요구사항을 새로 만들거나 바꾸는가" 하나다.** PRD는 요구사항
문서이므로, 요구사항이 그대로면 쓸 내용이 없다. 요구사항 불변 → 면제: 문서(`docs`), 리팩터링,
버그 수정(**기존** 요구사항의 복구 — 무엇이 옳은 동작인지는 이미 정의돼 있음), 성능 개선(요구사항
불변, 단 성능 목표 자체를 새로 세우면 요구사항 신설), 설정/의존성 갱신 중 요구사항에 안 닿는 것
(보안 패치 버전업·포맷). 반대로 기능 플래그 on·동작이 달라지는 업그레이드처럼 **설정·의존성이라도
요구사항을 바꾸면 면제가 아니다.** 면제로 판단하면 근거를 사용자에게 한 줄 알린다.
**새 기능·새 API·기존 동작의 의도적 변경 = 요구사항 변경 = 언제나 PRD 필수.**
(이 절이 면제 기준의 단일 정본 — 스킬·에이전트는 이 절을 참조한다)

## Skills — 특정 워크플로우

- **prd-writer** — PRD(제품 요구사항 문서) 생성 (`docs/prd/*.md`), **티켓·스펙보다 선행 · 필수 게이트**
  트리거: "PRD 만들어줘", "요구사항 문서 정리해줘", "개발 전에 문서부터"
- **spec-writer** — 개발 스펙 문서 생성 (`docs/MSG-XXX.md`)
  트리거: "MSG-XX 스펙 만들어줘", "스펙 문서 정리해줘"
- **spec-driven-dev** — 스펙 기반 개발 (ai-developer/convention-reviewer 팀 조율)
  트리거: "MSG-XX 개발 시작", "스펙대로 개발해줘", "MSG-XX 이어서/다시 개발"

## 하네스: FillMap-AI 개발 에이전트 팀

**목표:** MSG-XX 티켓 → PRD → 스펙 문서 → 파이썬 구현 → 컨벤션/검증까지 에이전트 팀
(spec-writer, ai-developer, convention-reviewer)이 처리.
에이전트 정의: `.claude/agents/`, 오케스트레이터: `.claude/skills/spec-driven-dev/`.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-07-31 | 초기 구성 — BE 하네스 이식(범용 프로세스 복사)+파이썬판 재구성(에이전트 3종·컨벤션), MSG-267 | 전체 | BE에서 검증된 파이프라인을 레포 경계 너머로 통일 |

## 알아둘 함정 (실측으로 배운 것)

- **번호판 가중치는 파일명을 명시할 것** — 레포에 n/s/m/l/x 5종이 같은 길이 이름으로 있어
  추측하면 조용히 large를 집는다. `bench.py`의 `PLATE_MODEL` 상수가 정본
- **macOS에서 스레드 제한으로 저사양 흉내 내지 말 것** — Apple Silicon PyTorch가
  torch 스레드 설정을 무시한다. 인스턴스 성능은 EC2 실측만 유효 (`scripts/ec2-bench.sh`)
- 프레임당 추론은 해상도 무관(`imgsz=640` 리사이즈) — 4K가 느린 건 프레임 수와 인코딩 탓.
  파이프라인 첫 단계는 **1080p 30fps 다운스케일** (MSG-143 전제 조건)
- **프레임 스킵·OpenVINO로 속도 내려 하지 말 것** — 실측 결과(MSG-207) OpenVINO는 +5%뿐이고,
  프레임 스킵은 도로 씬 번호판 커버리지가 46%로 붕괴한다(접근 차량 박스가 프레임당 수십 px 이동).
  건당 ~3분은 t3.small의 정직한 비용 — 속도가 필요하면 `results/MSG-207-report.md`의 다음 레버 순서로
- **`imgsz`는 640이 최적점 — 위아래로 다 손해다.** 상향(960·1280)은 recall이 오히려 떨어지고
  2~4배 느리며(MSG-158), 하향(480·320)은 작은 얼굴 64~91%·번호판 38~96%를 놓친다(MSG-281).
  두 모델 다 640으로 학습됐다. **더 만질 축이 아니다**
- **"추론 커널만 빠르게"는 전처리에 막힌다** — imgsz 320은 연산량이 1/4인데 실측 1.2~1.5배뿐이다.
  `predict()` 시간에는 1080p letterbox 리사이즈와 NMS가 섞여 있고 이건 imgsz와 무관하다.
  OpenVINO가 +5%였던 것(MSG-207)과 같은 벽이다
- **t3.small에서 배치 추론을 켜지 말 것** — MSG-339 단일 3회 중앙값은 배치 1이 180.53초,
  배치 2가 188.37초(+4.34%), 배치 4가 187.64초(+3.94%)였다. 검출 상자는 같았지만 속도
  기준을 못 넘겨 워커 2 실험도 게이트에서 중단했다. 배포값과 롤백값은 `AI_WORKERS=1`,
  `AI_BATCH_SIZE=1`이다
- **블러 커널은 ROI 긴 변 기준이다** — 짧은 변으로 잡으면 번호판처럼 납작한 박스에서 커널이
  글자 굵기보다 작아져 "가린 척"만 한다(MSG-280). 정상 완료로 보고되므로 미탐지보다 위험하다
- **블러 검증을 얼굴로만 하지 말 것** — 얼굴 박스는 정사각형에 가까워 위 함정을 비껴간다.
  게다가 커밋된 샘플 4종이 전부 Pexels 서양 영상이고, 유일한 번호판 샘플(`plates.mp4`)은
  원본 화질이 낮아 번호가 원래 안 읽힌다 — **번호판 결함이 드러나지 않는 조합**이었다.
  한국 번호판 검증은 별도 클립이 필요하다 (출처는 `results/MSG-280-blur.json`)
- **성능 요건을 절대 초로, 그것도 macOS 측정으로 잡지 말 것** — "실측은 EC2에서"는 다들
  지키면서 *요건을 세울 때* 이걸 어긴다. MSG-284가 판정 2초·탈락 5초를 macOS 기준으로 잡았다가
  EC2에서 7초·32초로 각각 2배·6배 미달이었고, **상대 기준**(정상 잡의 1/5 이하)으로 재설정했다.
  절대 초는 인스턴스 사양에 종속된다 — 성능 요건은 비율로 쓴다
- **`grab()`은 디코딩을 생략하지 않는다** — BGR 변환만 건너뛴다(실측 0.556 vs 0.993 ms/frame).
  그래서 순차 샘플링은 **영상 길이에 비례**하고 시킹은 평평하다. MSG-284가 시킹→순차로 바꿔
  EC2 판정 7.08→4.23초를 얻었지만 **교차점이 약 1분**이다(30초 0.46s vs 2분 1.82s).
  근거는 BE의 `durationSec ≤ 30` 검증 — **길이 상한이 풀리면 이 선택이 역행이 된다**
- **rawvideo 파이프에 `-shortest`를 단독으로 걸지 말 것** — 오디오가 영상보다 짧으면 ffmpeg가
  오디오 EOF에 파이프를 닫고 **exit 0**으로 끝나, 절단된 재생본이 정상 완료로 위장한다(MSG-367
  Codex 리뷰에서 발견, 재현 90/180프레임). `-af apad`를 함께 걸어야 기준이 항상 영상이 된다.
  같은 파이프의 stderr는 PIPE로 받지 말 것 — 사후 일괄 읽기는 64KB 버퍼가 차면 상호 블록이고,
  `AI_WORKERS=1`이라 잡 하나가 큐 전체를 영구 정지시킨다(임시 파일로 수거)

## 현재 상태 · 다음 작업

- 완료: MSG-142(측정) · MSG-143(ADR) · MSG-144(모델 선정) ·
  MSG-158(얼굴 conf 0.05 — 근거는 `results/MSG-158-report.md`) ·
  MSG-159(하이라이트 균등 3분할 폴백) · MSG-161(FastAPI 서버, dev EC2 Docker 상주) ·
  MSG-168(BE↔AI dev E2E 개통 — BE `AI_ENABLED=true` 상시 활성) ·
  MSG-151(부하 측정 — **Kafka 불필요** 판단, 근거는 `results/MSG-151-report.md`) ·
  MSG-280(번호판 블러 강도 수정 — 커널 긴 변 기준, 근거는 `results/MSG-280-report.md`) ·
  MSG-282(dev CD — main 머지 시 자동 배포) ·
  MSG-281(imgsz 하향 **기각** — recall 붕괴, 근거는 `results/MSG-281-report.md`) ·
  MSG-339(AI 전용 t3.small 분리 완료, 배치 2·4와 워커 2 채택 기각 — 근거는
  `results/MSG-339-report.md`) ·
  MSG-284(무의미 영상 프리체크 — grayscale std 중앙값 10 미만 탈락, 근거는 `results/MSG-284-report.md`.
  **BE 대응(MSG-286) 전까지 탈락 잡은 BE에서 PT30M 뒤에야 FAILED로 수렴한다** — 409를 "완료 전"으로
  해석해 재시도하기 때문. 큐 절감은 유효하고 손해는 그 사용자의 대기 시간뿐이다) ·
  MSG-353(하이라이트 선분석 `POST /highlights` — 블러 없이 동기 계산, 워커 큐 우회. 구간 품질
  보정(각 5초 이상·시작점 간격 5초, `HIGHLIGHT_MIN_SEC`)이 `/jobs` 저장 구간에도 적용.
  EC2 실측: 유휴 29초 5.3초·4K 18.5초·174초 31.9초, 블러 경합 시 3배 — 근거는
  `results/MSG-353-report.md`) ·
  MSG-367(AI 레포 안 전체 프레임 인코딩 3회→1회 — 무필터 다운스케일 바이패스 + `run()`의
  cv2.VideoWriter를 ffmpeg rawvideo 파이프로 교체해 mp4v 경유·병합 패스 제거. 재생본이
  4세대→3세대 인코딩본. 실영상 3종 전후 대조 detections·프레임 수 동일 —
  `results/MSG-367-identity.json`. EC2 전후 실측: 유휴 왕복 중앙값 221→192초(-13.1%),
  결과물 -21.9%, 메모리 동등 — 근거는 `results/MSG-367-report.md`) ·
  MSG-458(AI 경로 추천의 언어 양끝 — 동기 `POST /route/parse`(자연어+뷰포트 → 지역·기간·
  관심사·선호 순서)와 `POST /route/explain`(지점별 이유 한 줄), 잡 큐 우회. OpenAI
  `gpt-4o-mini` httpx 직호출(SDK 없음), pydantic strict 형태 방어, 인젝션 방어 3층.
  후보 선정·순서 배열은 BE(MSG-457) 몫. `ROUTE_AI_ENABLED` 기본 꺼짐, 스펙 `docs/MSG-458.md`.
  EC2 실측 완료 — 인젝션 6/6 방어, HTTP 왕복 중앙값 parse 0.9~1.2초·explain 1.1초, 블러 경합
  영향 없음. 근거는 `results/MSG-458-report.md`)
- 남은 것: 감지 건수(얼굴 N·번호판 N) 응답 추가(MSG-140 잔여 완료 조건) ·
  AI 처리량 확장은 실제 수요가 시간당 20건 이상으로 유지되면 인스턴스 유형부터 재평가 ·
  **MSG-284 탈락률 관측** — 운영 후 무의미 영상이 업로드의 2% 미만이면 프리체크가 손해다
  (판정 비용 > 절감. 손익분기 계산은 `results/MSG-284-report.md`). FR-7 판정 로그가 데이터 장치다
