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
