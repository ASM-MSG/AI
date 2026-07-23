# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

FillMap의 AI Highlight-Blur 서버. 얼굴·번호판 자동 블러 + 하이라이트 추천.
백엔드(Spring Boot, [ASM-MSG/BE](https://github.com/ASM-MSG/BE))와는 **HTTP로만 통신하는 별도 프로세스**다.

- 스택: Python · ultralytics(YOLOv11n) · PySceneDetect · ffmpeg · (예정) FastAPI
- 실행 환경: dev EC2에 Docker 컨테이너 상시 서버 — BE 레포 `docs/MSG-143.md` (ADR) 확정
- 모델 선정 근거: BE 레포 `docs/MSG-144.md`
- 측정 데이터: `results/MSG-142-report.md` — 처리 시간·메모리·발견된 문제 전부 여기

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

## 알아둘 함정 (실측으로 배운 것)

- **번호판 가중치는 파일명을 명시할 것** — 레포에 n/s/m/l/x 5종이 같은 길이 이름으로 있어
  추측하면 조용히 large를 집는다. `bench.py`의 `PLATE_MODEL` 상수가 정본
- **macOS에서 스레드 제한으로 저사양 흉내 내지 말 것** — Apple Silicon PyTorch가
  torch 스레드 설정을 무시한다. 인스턴스 성능은 EC2 실측만 유효 (`scripts/ec2-bench.sh`)
- 프레임당 추론은 해상도 무관(`imgsz=640` 리사이즈) — 4K가 느린 건 프레임 수와 인코딩 탓.
  파이프라인 첫 단계는 **1080p 30fps 다운스케일** (MSG-143 전제 조건)

## 현재 상태 · 다음 작업

- 완료: MSG-142(측정) · MSG-143(ADR) · MSG-144(모델 선정) ·
  MSG-158(얼굴 conf 0.05 — 근거는 `results/MSG-158-report.md`) ·
  MSG-159(하이라이트 균등 3분할 폴백) · MSG-161(FastAPI 서버, dev EC2 Docker 상주) ·
  MSG-168(BE↔AI dev E2E 개통 — BE `AI_ENABLED=true` 상시 활성) ·
  MSG-151(부하 측정 — **Kafka 불필요** 판단, 근거는 `results/MSG-151-report.md`)
- 남은 것: 감지 건수(얼굴 N·번호판 N) 응답 추가(MSG-140 잔여 완료 조건) ·
  AI 처리량 확장은 시간당 20건+ 지속 시 재평가(MSG-151 트리거)
