# AGENTS.md

FillMap AI 서버에서 일하는 AI 코딩 도구를 위한 진입점이다. Claude Code는 `CLAUDE.md`를 자동으로
읽지만 다른 도구(Codex 등)는 이 파일을 읽으므로, 여기서 실제 규칙 문서로 안내한다.

**규칙 본문은 여기에 적지 않는다.** 같은 규칙이 두 곳에 있으면 한쪽이 반드시 낡는다.
`CLAUDE.md`와 `.claude/rules/`가 정본이고, 이 파일은 어디를 봐야 하는지와 모르면 사고가 나는
것만 짚는다.

## 이 레포가 뭔가

FillMap의 AI Highlight-Blur 서버다. 영상에서 얼굴과 번호판을 찾아 자동으로 블러 처리하고
하이라이트 구간을 추천한다. 백엔드([ASM-MSG/BE](https://github.com/ASM-MSG/BE), Spring Boot)와는
**HTTP로만 통신하는 별도 프로세스**다.

스택은 Python · ultralytics(YOLOv11n) · PySceneDetect · ffmpeg · FastAPI.
dev EC2에 Docker 컨테이너로 상주하고, `main` 머지 시 자동 배포된다.

## 모르면 사고 나는 규칙 둘

### 1. 이 레포는 AGPL-3.0이다

Ultralytics 때문에 전염된다. **BE는 MIT를 유지해야 한다.**

- BE 코드와의 통신은 **HTTP만**. 이 레포의 코드를 BE에 복사하거나 import 하지 않는다
- 새 의존성은 라이선스를 먼저 확인한다. AGPL·GPL은 여기서는 괜찮지만 BE로 새어나가면 안 된다
- `samples/`는 gitignore 대상이다. 실제 얼굴·번호판이 담긴 영상은 커밋하지 않는다

### 2. macOS 실측은 무효다

Apple Silicon PyTorch가 torch 스레드 설정을 무시한다. 성능과 리소스 수치는 **dev EC2**
(`scripts/ec2-bench.sh`)에서만 잰다.

측정할 때만이 아니라 **요건을 세울 때도** 그렇다. MSG-284가 판정 2초·탈락 5초를 macOS 기준으로
잡았다가 EC2에서 7초·32초로 각각 2배·6배 미달이었다. 절대 초는 인스턴스 사양에 종속되므로
**성능 요건은 비율로 쓴다**(예: 정상 잡의 1/5 이하).

## 작업 전에 읽을 것

| 문서 | 내용 |
|---|---|
| `CLAUDE.md` | 전체 안내. 라이선스 규칙, 브랜치·커밋, PRD 게이트, **알아둘 함정** 목록 |
| `.claude/rules/coding-principles.md` | 코딩 행동 원칙 |
| `.claude/rules/project-conventions.md` | 파이썬 컨벤션(하드탭), 검증 관행, 의존성 추가 기준 |
| `.claude/rules/subagent-orchestration.md` | 에이전트 팀 운영 원칙 |

`CLAUDE.md`의 **"알아둘 함정"** 절은 전부 실측으로 배운 것이다. 성능을 개선하거나 파라미터를
만지기 전에 반드시 읽는다. 이미 기각된 시도가 여럿 적혀 있다(프레임 스킵, OpenVINO, imgsz 상하향).

`docs/MSG-XXX.md`는 티켓별 개발 스펙, `docs/prd/*.md`는 그 앞단의 제품 요구사항 문서,
`results/MSG-XXX-report.md`는 실측 리포트다.

## 검증 — 테스트 프레임워크가 없다

pytest 같은 프레임워크를 쓰지 않는다. 세 층으로 검증한다.

1. **`--smoke` 셀프체크 (필수)** — 로직을 추가하면 그 파일의 `--smoke` 경로가 그것을 실제로
   태우게 만든다. `bench.py --smoke`는 합성 영상으로 파이프라인을, `server.py --smoke`는
   httpx TestClient로 API 왕복을 검증한다. **새 검증 경로가 필요해도 프레임워크를 도입하지 말고
   기존 `--smoke`를 확장한다.**
2. **실측 리포트** — 성능·정확도 주장은 `results/MSG-XXX-report.md`와 원자료 JSON으로 남긴다.
   숫자 없는 "빨라졌다"는 완료 근거가 아니다.
3. **EC2 실측** — 위 2번 규칙 참조.

기능 정확도는 **recall 우선**으로 판단한다. 과블러는 손해가 작지만 미탐지는 프라이버시 사고다.

## 자주 어기는 규칙 셋

정본은 `CLAUDE.md`다. 여기 적은 건 요약이다.

### 새 브랜치에서 시작한다

`main`에 직접 커밋하지 않는다. 문서 작업도 예외가 없다. git flow 타입만 쓰고 하이픈으로 잇는다.

```
feature/MSG-284-precheck    (O)
feature/MSG-284/precheck    (X) 슬래시
chore/MSG-284-precheck      (X) git flow에 없는 타입
```

기본 브랜치가 `main`이라 PR base도 `main`이다 (BE 레포는 `develop`이니 헷갈리지 않는다).

### 커밋은 제목 한 줄이 기본이다

형식은 `MSG-{번호} {타입}: {요약}`. 타입은 feat·fix·refactor·docs·test·chore·style.
제목은 코드를 안 연 사람이 무엇이 바뀌었는지 아는 문장이어야 한다. 가운뎃점 나열은 쓰지 않는다.
본문은 제목으로 설명이 안 되는 근거(실측 수치, 기각한 대안)가 있을 때만 붙인다.

### PRD가 스펙과 구현보다 앞선다

```
아이디어/티켓 → PRD(docs/prd/*.md) → 스펙(docs/MSG-XXX.md) → 구현
                 ↑ 필수 게이트
```

면제 기준은 하나다. **이 작업이 제품 요구사항을 새로 만들거나 바꾸는가.**
요구사항이 그대로면 면제다(문서, 리팩터링, 버그 수정, 성능 개선). 새 기능·새 API·기존 동작의
의도적 변경은 언제나 PRD가 필요하다. 판단 기준의 정본은 `CLAUDE.md`의 "개발 파이프라인" 절이다.

### PR 본문은 템플릿을 그대로 채운다

`.github/PULL_REQUEST_TEMPLATE.md`의 네 절(관련 티켓 / 작업 내용 / 고민한 내용 / 리뷰 포인트)을
그 순서대로 쓴다.

## 실행

```bash
python server.py --smoke     # API 왕복 셀프체크
python bench.py --smoke      # 파이프라인 셀프체크 (합성 영상)
scripts/ec2-bench.sh         # dev EC2 실측 (성능 수치는 여기서만)
```

각 스크립트의 실행법은 파일 상단 docstring에 티켓 번호와 함께 적혀 있다.

## 도구별 참고

`.claude/agents/`의 에이전트 정의(`ai-developer`, `spec-writer`, `convention-reviewer`)와
`spec-driven-dev` 워크플로는 Claude Code의 서브에이전트 기능에 의존한다. 다른 도구에서는 그대로
쓸 수 없으니 절차 설명서로 읽고 수동으로 따른다.
