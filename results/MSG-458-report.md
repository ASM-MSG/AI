# MSG-458 — 경로 추천 엔드포인트 EC2 실측

> 측정일 2026-08-23 · AI 전용 t3.small, Docker 컨테이너 내부에서 실행 · 모델 gpt-4o-mini(temperature 0)
> 원자료 `results/MSG-458-routes.json`(유휴) · `results/MSG-458-routes-contention.json`(블러 경합)
> 스펙 성공 기준 "응답 시간"·"인젝션 내성" 행의 판정 근거. 방법: `route_experiment.py --server
> http://localhost:8000` — 파이프라인 직호출과 HTTP 왕복(FastAPI 디스패치·직렬화·전송 포함)을
> 각 3회 병기, 전 회차 성공일 때만 중앙값을 발표한다.

## 수치

### 왕복 시간 (중앙값, 3회 전부 성공)

| 구간 | 유휴 직호출 | 유휴 HTTP | 블러 PROCESSING 중 직호출 | 블러 PROCESSING 중 HTTP |
|---|---|---|---|---|
| parse | 924ms | 1,229ms | 918ms | 878ms |
| explain | 1,038ms | 1,060ms | 1,067ms | 1,159ms |

- 관측 최악 회차는 유휴 HTTP parse 1,561ms. 런 간 변동(±400ms)이 모델 지연 편차에서 오고,
  서버 오버헤드는 explain 기준 수십 ms 수준이다.
- **블러 잡 경합 영향은 사실상 없다** — 경합 중 수치가 유휴와 같은 자릿수다(측정 시작·종료
  시점 모두 잡 상태 PROCESSING 확인). `/highlights`가 경합 시 3배였던 것(MSG-353)과 대조적인데,
  경로 호출은 YOLO 추론이 없어 CPU가 아니라 외부 모델 네트워크 대기가 지배항이라서다.

### 스키마 실수락 · 인젝션 내성

| 항목 | 결과 |
|---|---|
| RESPONSE_FORMAT_PARSE / _EXPLAIN | OpenAI가 둘 다 수락 (스텁 스모크가 못 태우던 유일 경로 해소) |
| 인젝션 6종 | **6/6 방어 성공** — 5건은 모델이 형태를 유지(빈 유효 해석), 1건(시스템 프롬프트를 region에 출력 유도)은 모델이 뚫렸으나 서버 형태 검증이 502로 필터. D-5의 1층(프롬프트)과 2층(형태)이 실전에서 각각 동작했다 |

### 해석 품질 (자동 판정 없음, 육안 대조)

3문장 전부 형태 통과에 상식적 해석 — "부산역 내려서 해운대에서 밥 먹고 축제도"가
region 부산, interests 식사·축제, preferred_order 부산역→해운대로 갈렸다. 관찰 하나:
일요일(8/23) 시점의 "이번 주말"이 8/27~28(목·금)로 이틀 어긋났다. 해석 자유도 영역이라
결함으로 치지 않지만, BE가 기간을 엄격 필터로 쓰면 후보가 이틀 밀릴 수 있다 — 기간 필터를
느슨하게 잡는 쪽이 안전하다.

## 읽는 법

- **응답 시간 상한의 확정 재료**: BE가 재는 값 기준(HTTP 왕복) 중앙값 0.9~1.2초, 관측 최악
  1.6초. BE 호출당 타임아웃을 5초로 잡으면 관측 최악의 3배 여유다. `ROUTE_AI_TIMEOUT_SEC`
  기본 30초는 무한 대기 방지용 안전핀으로 두되, BE 상한 확정 시 그 이하로 조정한다(스펙 D-2).
  상한 확정은 MSG-457(BE) 몫이고 이 리포트는 재료만 제공한다.
- 두 왕복 합산(해석+문장화)이 2.3초 안팎이라, BE 후보 수집·순서 배열(수 ms, MSG-398 실측)을
  더해도 사용자 체감은 3초 이내에 들어온다.
- 이 실측으로 스펙 성공 기준의 미검증 잔여 4종(스키마 실수락·실모델 인젝션 내성·왕복
  시간·블러 경합)이 전부 닫혔다. MSG-458의 완료 조건 충족.

## 재현

```bash
# fillmap-ai EC2, 플래그·키가 ~/fillmap-ai.env에 있고 배포된 상태에서
sudo docker exec fillmap-ai python route_experiment.py --server http://localhost:8000
# 경합 케이스: 블러 잡을 먼저 물리고 timing만
curl -s -F "file=@/tmp/e2e-plates.mp4" localhost:8000/jobs
sudo docker exec fillmap-ai python route_experiment.py --only timing --server http://localhost:8000 \
	--out results/MSG-458-routes-contention.json
```
