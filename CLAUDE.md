# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

FillMap의 AI Highlight-Blur 서버. 얼굴·번호판 자동 블러 + 하이라이트 추천.
백엔드(Spring Boot, [ASM-MSG/BE](https://github.com/ASM-MSG/BE))와는 **HTTP로만 통신하는 별도 프로세스**다.

- 스택: Python · ultralytics(YOLOv11n) · PySceneDetect · ffmpeg · (예정) FastAPI
- 실행 환경: dev EC2에 Docker 컨테이너 상시 서버 — BE 레포 `docs/MSG-143.md` (ADR) 확정
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

## 현재 상태 · 다음 작업

- 완료: MSG-142(측정) · MSG-143(ADR) · MSG-144(모델 선정) ·
  MSG-158(얼굴 conf 0.05 — 근거는 `results/MSG-158-report.md`) ·
  MSG-159(하이라이트 균등 3분할 폴백) · MSG-161(FastAPI 서버, dev EC2 Docker 상주) ·
  MSG-168(BE↔AI dev E2E 개통 — BE `AI_ENABLED=true` 상시 활성) ·
  MSG-151(부하 측정 — **Kafka 불필요** 판단, 근거는 `results/MSG-151-report.md`)
- 남은 것: 감지 건수(얼굴 N·번호판 N) 응답 추가(MSG-140 잔여 완료 조건) ·
  AI 처리량 확장은 시간당 20건+ 지속 시 재평가(MSG-151 트리거)
