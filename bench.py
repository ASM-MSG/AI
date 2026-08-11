"""MSG-142 — 블러·하이라이트 파이프라인 처리 시간·리소스 측정.

병목이 AI인지 인코딩인지 가르는 게 목적이라 단계별로 시간을 따로 누적한다.
정확도 튜닝은 범위 밖 (MSG-142 티켓 명시).

MSG-367: 출력은 mp4v 중간본이 아니라 최종 재생본(h264 + 원본 오디오)이다 — 블러 프레임을
rawvideo 파이프로 ffmpeg에 흘려 단일 패스로 굽는다. 목적은 세대 손실 제거다 (docs/MSG-367.md).
`stages.encode`의 의미도 "mp4v 기록"에서 "파이프 쓰기 + 인코더 배압 대기"로 바뀌었다 —
변경 전 리포트의 stage 비중과 직접 비교하지 말 것.

    python bench.py --smoke                        # 합성 영상으로 파이프라인 검증
    python bench.py sample.mp4 --device cpu
    python bench.py sample.mp4 --device mps        # Apple Silicon
    python bench.py sample.mp4 --device cuda       # AWS GPU 인스턴스

MSG-207 실험 노브 (환경변수) — **둘 다 실측 결과 미채택, 기본값이 정본** (results/MSG-207-report.md):
    AI_BACKEND=openvino       # +5%뿐이라 미채택. 쓰려면 openvino 별도 설치 필요
    AI_INFER_STRIDE=3         # 도로 씬 번호판 커버 46%로 붕괴 — 프라이버시 사고라 미채택
    AI_BOX_PAD=0.10           # 스킵 프레임 재사용 박스 확장 비율 (stride>1일 때만 의미)

MSG-339 배치 노브 (전용 EC2 실측 전 기본값 1):
    AI_BATCH_SIZE=1           # 허용값 1·2·4. 채택값은 results/MSG-339-report.md에서 확정
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# (repo, 가중치 파일)을 못박는다. 번호판 레포는 n/s/m/l/x 5종을 같은 길이 이름으로
# 올려둬서 파일명을 추측하면 조용히 large를 집는다 — nano를 명시할 것.
FACE_MODEL = ("AdamCodd/YOLOv11n-face-detection", "model.pt")
PLATE_MODEL = ("morsetechlab/yolov11-license-plate-detection", "license-plate-finetune-v1n.pt")

# 블러는 과하게 걸려도 손해가 작지만 미탐지는 프라이버시 사고다 → 기본값을 낮게 잡는다.
# 얼굴 0.05는 MSG-158 실험 결과 (results/MSG-158-report.md) — recall 0.72→0.98, 시간 불변.
# 번호판은 미실험이라 0.25 유지.
FACE_CONF = 0.05
PLATE_CONF = 0.25

# MSG-284: 프리체크 — grayscale std 중앙값이 이 값 미만이면 탈락(암흑·렌즈 가림).
# 근거는 results/MSG-284-report.md — PASS 최소 36.69(night-dark) / FAIL 최대 3.18(dark-covered) 사이.
# 오탐(정상 영상 탈락)이 사고이고 미탐은 3분 낭비로 끝난다 → PASS 쪽에 넉넉히 붙였다.
PRECHECK_STD_THRESHOLD = 10.0
PRECHECK_FRAMES = 10   # 20/10/5장에서 std가 ±0.2% 이내라 20장은 낭비고, 5장은 중앙값 표본이 너무 작다

# 밝은 픽셀로 칠 문턱. 판정에는 안 쓰고 bright_pct 로깅용이다 —
# "몇 이상을 밝다고 칠 것인가"가 자의적이라 임계값 지표로는 채택하지 않았다 (MSG-284 리포트)
BRIGHT_LEVEL = 64

# MSG-353: 하이라이트 구간 품질 — 후보끼리 절반 이상 겹치면 FE 선택이 무의미해진다.
# 값의 정본은 BE 레포 docs/prd/MSG-351-prd.md FR-3 (구간 최소 5초, 시작점 간격 5초).
HIGHLIGHT_MIN_SEC = 5.0
HIGHLIGHT_MIN_GAP_SEC = 5.0

# MSG-367: stdin을 닫은 뒤 ffmpeg 잔여 플러시 대기 상한(초). 배압 구조상 파이프에 남는 건
# 프레임 몇 장이라 정상 플러시는 수 초다 — 넘기면 행으로 보고 kill해 워커를 살린다.
ENCODER_FLUSH_TIMEOUT_SEC = 60

# MSG-207: 추론 백엔드·프레임 스트라이드. 실측 결과 두 노브 모두 미채택 — 기본값이 정본.
BACKEND = os.environ.get("AI_BACKEND", "torch")             # torch | openvino
INFER_STRIDE = int(os.environ.get("AI_INFER_STRIDE", "1"))  # k프레임마다 1회 추론
BOX_PAD = float(os.environ.get("AI_BOX_PAD", "0.10"))       # 재사용 박스 확장 비율
AI_BATCH_SIZE = int(os.environ.get("AI_BATCH_SIZE", "1"))
if AI_BATCH_SIZE not in (1, 2, 4):
	raise ValueError("AI_BATCH_SIZE는 1, 2, 4만 허용")


class Stage:
	"""단계별 누적 시간. with 블록 하나가 한 번의 호출."""

	def __init__(self):
		self.totals = {}

	def track(self, name):
		return _Timer(self.totals, name)

	def report(self, wall):
		rows = sorted(self.totals.items(), key=lambda kv: -kv[1])
		return {name: {"sec": round(t, 3), "pct": round(100 * t / wall, 1)} for name, t in rows}


class _Timer:
	def __init__(self, totals, name):
		self.totals = totals
		self.name = name

	def __enter__(self):
		self.start = time.perf_counter()

	def __exit__(self, *exc):
		self.totals[self.name] = self.totals.get(self.name, 0.0) + (time.perf_counter() - self.start)


def peak_memory_mb():
	"""ru_maxrss 단위가 OS마다 다르다 — Linux는 KB, macOS는 byte."""
	peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return round(peak / (1024 if sys.platform == "linux" else 1024 * 1024), 1)


def load_models(device, backend=None):
	from huggingface_hub import hf_hub_download
	from ultralytics import YOLO

	backend = backend or BACKEND

	def load(repo, filename):
		pt = hf_hub_download(repo, filename)
		if backend == "openvino":
			# MSG-207: 같은 가중치의 OpenVINO 변환 — x86 CPU에서 추론 가속.
			# export 산출물은 가중치 옆(HF 캐시)에 남아 hf-cache 볼륨과 함께 재시작에도 유지된다.
			ov = Path(pt).with_name(Path(pt).stem + "_openvino_model")
			if not ov.exists():
				YOLO(pt).export(format="openvino")
			return YOLO(str(ov), task="detect")
		model = YOLO(pt)
		model.to(device)
		return model

	return load(*FACE_MODEL), load(*PLATE_MODEL)


def blur_boxes(frame, boxes):
	"""검출 영역만 잘라 가우시안 블러 후 되붙인다."""
	for x1, y1, x2, y2 in boxes:
		roi = frame[y1:y2, x1:x2]
		if roi.size == 0:
			continue
		# 커널을 ROI 크기에 비례시켜야 작은 얼굴도 실제로 뭉개진다.
		# 기준은 반드시 **긴 변**이다 — 짧은 변으로 잡으면 번호판처럼 납작한 박스에서
		# 커널이 글자 굵기보다 작아져 "가린 척"만 하고 번호가 그대로 읽힌다 (MSG-280).
		k = max(3, (max(roi.shape[:2]) // 4) | 1)
		frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
	return frame


def to_boxes(result, width, height):
	out = []
	for box in result.boxes.xyxy.tolist():
		x1, y1, x2, y2 = (int(v) for v in box)
		out.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
	return out


def pad_boxes(boxes, width, height, pad=None):
	"""스킵 프레임에 재사용할 박스는 프레임 간 이동을 흡수하도록 확장한다 (MSG-207)."""
	pad = BOX_PAD if pad is None else pad
	out = []
	for x1, y1, x2, y2 in boxes:
		dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
		out.append((max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)))
	return out


def detect_highlights(path, duration, stage):
	"""장면 랭킹 상위 최대 3구간 — 각 5초 이상, 시작점 간격 5초 이상 (MSG-353). 5초 미만이면 빈 배열 (MSG-141)."""
	if duration < HIGHLIGHT_MIN_SEC:
		return []
	from scenedetect import ContentDetector, detect

	with stage.track("highlight"):
		scenes = detect(str(path), ContentDetector())
	# 장면을 5초 이상 창으로 정규화한 랭킹 후보 뒤에 균등 그리드 보충 후보를 세우고,
	# 시작점 간격 5초를 지키며 앞에서부터 탐욕 선택한다. 장면 0개(고정 촬영)면 그리드만 남는다 —
	# 기존 균등 3분할(MSG-159)은 12초 영상에서 4초짜리 구간 3개를 만들어 폐기했다 (docs/MSG-353.md D-3)
	ranked = sorted(scenes, key=lambda s: (s[1] - s[0]).get_seconds(), reverse=True)
	candidates = []
	for s, e in ranked:
		start, end = s.get_seconds(), e.get_seconds()
		if end - start < HIGHLIGHT_MIN_SEC:
			# 짧은 장면은 그 시작점부터 5초 창으로 넓힌다 — 영상 끝을 넘으면 시작점을 앞으로 당긴다
			start = max(0.0, min(start, duration - HIGHLIGHT_MIN_SEC))
			end = start + HIGHLIGHT_MIN_SEC
		candidates.append((start, end))
	grid_count = min(3, int(duration // HIGHLIGHT_MIN_SEC))
	# duration >= 5*grid_count 라 균등 배치 간격이 수학적으로 5초 이상 보장된다 (docs/MSG-353.md D-3)
	step = (duration - HIGHLIGHT_MIN_SEC) / (grid_count - 1) if grid_count > 1 else 0.0
	candidates += [(i * step, i * step + HIGHLIGHT_MIN_SEC) for i in range(grid_count)]

	picked = []
	for start, end in candidates:
		if len(picked) == 3:
			break
		if all(abs(start - p[0]) >= HIGHLIGHT_MIN_GAP_SEC for p in picked):
			picked.append((start, end))
	return [[round(start, 2), round(end, 2)] for start, end in picked]


def sample_frames(path, n):
	"""영상에서 n프레임을 균등 간격으로 뽑는다. 뽑는 인덱스는 `int(i * total / n)`.

	**시킹(`cap.set(POS_FRAMES)`)을 쓰지 않는다.** 시킹은 목표 프레임마다 앞선 키프레임까지
	되감아 다시 디코딩하므로 같은 구간을 반복해서 푼다. 순차 `grab()`은 되감기가 없어
	dev EC2 실측 10프레임 확보 6.49초 → 3.81초.

	**단, `grab()`은 디코딩을 생략하지 않는다** — BGR 변환(retrieve)만 건너뛴다
	(실측 0.556 vs 0.993 ms/frame). 즉 이 함수는 **O(총 프레임)** 이라 영상이 길수록 불리하고,
	시킹은 길이와 무관하게 평평하다. **교차점이 약 1분**이다(30초 0.46초 / 2분 1.82초).
	30초 이하가 전제이며 근거는 BE의 `durationSec <= 30` 검증이다 —
	**길이 상한이 풀리면 이 선택이 역행이 된다.** 길이별 실측: results/MSG-284-report.md
	"""
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise SystemExit(f"영상을 열 수 없음: {path}")
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	out = []
	pos = 0
	for i in range(n):
		want = int(i * total / n)
		if out and want == out[-1][0]:
			# n > total이면 같은 인덱스가 연속으로 나온다. 순차 재생은 되감을 수 없으니
			# 앞서 디코딩한 프레임을 그대로 재사용한다 — 시킹판과 결과가 같아야 한다.
			out.append(out[-1])
			continue
		while pos < want and cap.grab():
			pos += 1
		if pos < want:
			break  # 메타데이터의 total보다 실제가 짧다 — 남은 인덱스는 어차피 못 읽는다
		ok, frame = cap.read()
		if not ok:
			break
		pos += 1
		out.append((want, frame))
	cap.release()
	return out


def frame_metrics(frame):
	"""프레임 하나의 지표. 전부 grayscale 기준 — 색은 판정에 쓰지 않는다.

	판정은 std만 쓰지만 나머지도 남긴다 — 배포 후 실제 업로드로 임계값을 사후 조정할
	근거이고(MSG-284 FR-7), 10프레임 계산이라 비용이 무시할 수준이다.
	"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
	p = hist / hist.sum()
	nz = p[p > 0]
	return {
		"mean": float(gray.mean()),
		"std": float(gray.std()),
		"p99": float(np.percentile(gray, 99)),
		# 밝은 픽셀 비율(%). 밤이어도 조명은 있고, 가려진 렌즈엔 없다.
		"bright_pct": float(100 * (gray > BRIGHT_LEVEL).mean()),
		# 라플라시안 분산 = 엣지 강도. **판정에 쓰면 안 된다** — 센서 노이즈를 엣지로 오인해
		# 방향이 뒤집힌다(PASS 6.52 < FAIL 179.31, results/MSG-284-report.md)
		"lap_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
		# 히스토그램 엔트로피(bit). 한 밝기에 몰려 있으면 낮다.
		"entropy": float(-(nz * np.log2(nz)).sum()),
	}


def median_metrics(frames):
	"""프레임별 지표의 **중앙값**. 평균이 아닌 이유는 오탐 방지다 — 촬영 시작 직후만 가려진
	영상이 전체 탈락하면 안 된다. 소수 프레임이 아니라 영상 전반이 어두워야 탈락이다."""
	per_frame = [frame_metrics(f) for _, f in frames]
	return {k: round(float(np.median([m[k] for m in per_frame])), 2) for k in per_frame[0]}


def video_metrics(path, n_frames):
	"""영상 단위 지표. 실험 스크립트가 쓰는 경로라 몇 장 실패는 관대하게 넘긴다."""
	frames = sample_frames(path, n_frames)
	if not frames:
		# 없으면 IndexError로 새는데, 원인이 안 드러나는 실패는 FR-8 취지에 어긋난다
		raise SystemExit(f"프레임을 읽을 수 없음: {path}")
	return median_metrics(frames)


def precheck(path, n_frames=PRECHECK_FRAMES):
	"""MSG-284: 추론 전 무의미 영상(암흑·렌즈 가림) 판정. 규칙 정본은 results/MSG-284-report.md.

	블러와 정확도 방향이 반대다 — **오탐이 사고, 미탐은 3분 낭비**라 애매하면 통과다.
	reason의 코드 접두어(too_dark)는 BE 매핑용으로 고정, 콜론 뒤 수치는 진단용이라 형식이 바뀔 수 있다.

	ponytail: 픽셀 통계 단일 지표 — 내용 기반 판정(책상·벽만 찍힌 밝은 영상)이 필요해지면 CLIP zero-shot (PRD §8)
	"""
	frames = sample_frames(path, n_frames)
	if len(frames) * 2 < n_frames:
		# 중앙값으로 판정하므로 표본이 절반 미만이면 판정 근거 자체가 부족하다.
		# 손상된 입력을 "너무 어두움"으로 오진하지 않고 기존 실패 경로로 보낸다 (FR-8).
		# '전부 요구'는 하지 않는다 — 메타데이터가 부정확한 정상 영상까지 FAILED가 되면 그게 오탐이다.
		raise SystemExit(f"프레임 디코딩 실패: {path} — {n_frames}장 중 {len(frames)}장만 읽힘")
	metrics = median_metrics(frames)
	std = metrics["std"]
	passed = std >= PRECHECK_STD_THRESHOLD
	return {
		"passed": passed,
		"reason": None if passed else f"too_dark: std {std:.2f} < {PRECHECK_STD_THRESHOLD}",
		"metrics": metrics,
	}


def run(path, device, out_path, audio_src=None):
	"""블러 + 하이라이트 + 단일 인코딩. out_path가 곧 최종 재생본이다 (MSG-367).

	audio_src=None이면 입력 path를 오디오 원본으로 쓴다 — 단독 CLI 실행도 무음 mp4v 대신
	오디오 있는 h264를 얻는다. server.process()는 다운스케일본에 오디오가 없어(-an)
	잡 원본을 명시한다 (docs/MSG-367.md D-1·D-3).
	"""
	stage = Stage()
	wall_start = time.perf_counter()

	with stage.track("model_load"):
		face, plate = load_models(device)
	if BACKEND == "openvino":
		device = "cpu(openvino)"

	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise SystemExit(f"영상을 열 수 없음: {path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# MSG-367 D-2·D-3: rawvideo 파이프 단일 인코딩. fps는 cv2가 읽은 소수 그대로 넘긴다 —
	# 반올림하면 A/V 싱크가 영상 길이에 비례해 밀린다. -map 1:a?의 ?가 오디오 없는 입력을
	# 에러 없이 무음 재생본으로 만든다(그때 -af apad는 조용히 무시된다 — 실측 확인).
	# -af apad는 오디오를 무음으로 영상 끝까지 채워 -shortest의 기준을 항상 영상으로 만든다 —
	# -shortest 단독은 오디오가 영상보다 짧은 정상 입력에서 오디오 EOF에 stdin을 닫고 exit 0으로
	# 끝나, 절단된 재생본이 정상 완료로 위장한다. 영상보다 긴 꼬리 오디오는 여전히 영상 끝에서
	# 잘린다(D-3 의도 불변). yuv420p 명시 — bgr24 입력을 그대로 두면 일부 재생기가 못 여는
	# 픽셀 포맷으로 인코딩될 수 있다.
	# stderr는 파이프가 아니라 임시 파일로 받는다 — 파이프는 손상 오디오가 에러를 쏟아 64KB
	# 버퍼가 차면 ffmpeg(stderr 쓰기)와 파이썬(stdin 쓰기)이 상호 블록되고, AI_WORKERS=1이라
	# 잡 하나가 큐 전체를 영구 정지시킨다 (D-4).
	stderr_file = tempfile.TemporaryFile()
	encoder = subprocess.Popen(
		["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
		 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
		 "-i", str(audio_src if audio_src is not None else path), "-map", "0:v", "-map", "1:a?", "-af", "apad",
		 # 홀수 해상도 방어 (Codex 라운드3) — 변경 전 동작은 입력 픽셀 포맷에 따라 갈렸다:
		 # 420p 홀수(실입력 대부분)는 패스 2 libx264에서 동일 실패, 444 등 홀수 허용 포맷은
		 # VideoWriter가 조용히 1px 크롭해 완주. 파이프판(bgr24→yuv420p 강제)은 양쪽 다 실패로
		 # 만들므로 pad가 크롭 아닌 1px 패딩으로 전부 완주하게 복원·개선한다(픽셀 유실 없음).
		 # BE 정규화 입력은 항상 짝수라 도달 경로는 CLI·직접 업로드뿐. 탐지·블러는 파이프 이전
		 # 원본 프레임에서 끝나므로 좌표 무관, 우/하단 1px 검은 테두리만 붙는다
		 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
		 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
		 str(out_path)],
		stdin=subprocess.PIPE, stderr=stderr_file)

	frames = face_hits = plate_hits = inference_batches = 0
	reused = []
	pipe_broken = False
	loop_ok = False  # 예외 경로 판별용 — finally에서 kill 선행 여부를 가른다 (Codex 라운드2)
	try:
		while not pipe_broken:
			pending = []
			while len(pending) < AI_BATCH_SIZE:
				with stage.track("decode"):
					ok, frame = cap.read()
				if not ok:
					break
				pending.append((frames, frame))
				frames += 1
			if not pending:
				break

			infer_frames = [frame for index, frame in pending if index % INFER_STRIDE == 0]
			face_results = plate_results = []
			if infer_frames:
				with stage.track("infer_face"):
					face_results = face.predict(infer_frames, conf=FACE_CONF, verbose=False)
				with stage.track("infer_plate"):
					plate_results = plate.predict(infer_frames, conf=PLATE_CONF, verbose=False)
				inference_batches += 1

			result_index = 0
			for index, frame in pending:
				if index % INFER_STRIDE == 0:
					fb = to_boxes(face_results[result_index], width, height)
					pb = to_boxes(plate_results[result_index], width, height)
					result_index += 1
					face_hits += len(fb)
					plate_hits += len(pb)
					boxes = fb + pb
					reused = pad_boxes(boxes, width, height)
				else:
					# MSG-207: 스킵 프레임은 직전 추론 박스의 확장본을 재사용한다.
					# 커버리지 근거는 results/MSG-207-report.md (stride·pad 시뮬레이션)
					boxes = reused

				with stage.track("mask"):
					frame = blur_boxes(frame, boxes)
				with stage.track("encode"):
					# 인코더가 밀리면 여기가 블록되는 자연 배압 — 그 대기가 encode 항목에 잡힌다 (MSG-367 D-2)
					try:
						encoder.stdin.write(frame.tobytes())
					except BrokenPipeError:
						# ffmpeg 조기 사망 (D-4) — 남은 프레임을 버리고 아래 반환 코드 검사로 실패를 드러낸다
						pipe_broken = True
						break
		loop_ok = True
	finally:
		# 추론·박스 변환·블러가 예외를 던져도 인코더를 회수한다 — 워커는 잡 간 지속되므로
		# 안 거두면 실패 잡마다 ffmpeg 프로세스와 fd가 누적된다 (D-4)
		cap.release()
		if not loop_ok:
			# 예외 경로는 kill 선행 (Codex 라운드2) — 인코더가 가득 찬 파이프에서 스톨해 있으면
			# close()의 잔여 버퍼 flush가 아래 wait(timeout)에 닿기 전에 영구 블록된다. 죽이고 나면
			# flush는 BrokenPipeError/OSError로 아래 가드에 잡히고, 이 경로의 출력은 어차피 버려진다.
			# 정상·pipe_broken 경로는 꼬리 프레임 flush가 필요하거나 이미 죽어 있어 kill하지 않는다
			encoder.kill()
		try:
			encoder.stdin.close()
		except (BrokenPipeError, OSError):
			# 마지막 버퍼 flush에서도 조기 사망이 드러날 수 있다 — 판정은 아래 반환 코드가 한다
			pass
		try:
			encoder.wait(timeout=ENCODER_FLUSH_TIMEOUT_SEC)
		except subprocess.TimeoutExpired:
			encoder.kill()
			encoder.wait()
		stderr_file.seek(0)
		stderr = stderr_file.read().decode(errors="replace")
		stderr_file.close()
	if encoder.returncode != 0:
		# cv2.VideoWriter는 실패해도 조용히 빈 파일을 남겼다 — 파이프판은 잡 FAILED로 드러낸다 (D-4).
		# 프레임 0장 입력도 ffmpeg가 빈 비디오로 실패해 여기로 온다 — 손상 입력이 성공으로 위장하던 구멍
		raise RuntimeError(f"ffmpeg 인코딩 실패 (exit {encoder.returncode}): {stderr.strip()}")

	duration = frames / fps if fps else 0
	highlights = detect_highlights(path, duration, stage) if duration >= 5 else []
	wall = time.perf_counter() - wall_start

	infer = stage.totals.get("infer_face", 0) + stage.totals.get("infer_plate", 0)
	return {
		"input": str(path),
		"device": device,
		"backend": BACKEND,
		"infer_stride": INFER_STRIDE,
		"batch_size": AI_BATCH_SIZE,
		"inference_batches": inference_batches,
		"resolution": f"{width}x{height}",
		"frames": frames,
		"video_sec": round(duration, 2),
		"wall_sec": round(wall, 2),
		"realtime_factor": round(wall / duration, 2) if duration else None,
		"ms_per_frame_inference": round(1000 * infer / frames, 2) if frames else None,
		"peak_memory_mb": peak_memory_mb(),
		"detections": {"face": face_hits, "plate": plate_hits},
		"highlights": highlights,
		"stages": stage.report(wall),
	}


def make_smoke_video(path, seconds=6, fps=30, size=(640, 480), audio_sec=None):
	"""검출 결과가 0건이어도 파이프라인은 완주해야 한다 (MSG-140 완료 조건).

	audio_sec는 lavfi 사인톤 길이(초), None이면 무음 — 재생본의 원본 오디오 유지 검증용 (MSG-367).
	영상보다 짧게 주면 -shortest가 영상을 절단하던 회귀(-af apad로 수정)를 검증하는 입력이 된다.
	"""
	args = ["-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size[0]}x{size[1]}:rate={fps}"]
	if audio_sec:
		args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={audio_sec}", "-c:a", "aac"]
	# 홀수 해상도 입력은 yuv444p로 만든다 — yuv420p+libx264는 홀수 폭/높이를 못 굽는다.
	# 444는 크로마 서브샘플링이 없어 홀수가 허용된다 (run()의 pad 방어 검증용 입력, MSG-367)
	pix_fmt = "yuv420p" if size[0] % 2 == 0 and size[1] % 2 == 0 else "yuv444p"
	subprocess.run(["ffmpeg", "-y", *args, "-pix_fmt", pix_fmt, str(path)], check=True, capture_output=True)


def ffprobe_codec(path, stream="v:0"):
	"""스트림 코덱명. 스트림이 없으면 빈 문자열 — mp4v 흔적·오디오 유무의 기계 판정용 (MSG-367)."""
	out = subprocess.run(
		["ffprobe", "-v", "error", "-select_streams", stream,
		 "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
		check=True, capture_output=True, text=True)
	return out.stdout.strip()


def make_dark_video(path, dark_sec=6, bright_sec=0, size=(640, 480), fps=30):
	"""검은 화면 dark_sec초 뒤에 컬러바 bright_sec초를 이어 붙인다 (MSG-284 프리체크 검증용).

	bright_sec=0이면 검은 화면 단독 = 렌즈 가림 근사라 **탈락**해야 하고,
	bright_sec>0은 "촬영 시작 직후에만 가려진 영상"이라 중앙값 규칙이면 **통과**해야 한다.
	"""
	wh = f"{size[0]}x{size[1]}"
	args = ["-f", "lavfi", "-i", f"color=c=black:s={wh}:r={fps}:d={dark_sec}"]
	if bright_sec:
		args += ["-f", "lavfi", "-i", f"testsrc=duration={bright_sec}:size={wh}:rate={fps}",
			"-filter_complex", "[0:v][1:v]concat=n=2:v=1"]
	subprocess.run(["ffmpeg", "-y", *args, "-pix_fmt", "yuv420p", str(path)],
		check=True, capture_output=True)


def smoke():
	with tempfile.TemporaryDirectory() as tmp:
		src, dst = Path(tmp) / "in.mp4", Path(tmp) / "out.mp4"
		make_smoke_video(src, seconds=6.1)  # 배치 4의 마지막 3장이 별도 추론 경로를 타야 한다 (MSG-339)
		report = run(src, "cpu", dst)

		assert report["frames"] > 0, "프레임을 하나도 읽지 못했다"
		batch_size = int(os.environ.get("AI_BATCH_SIZE", "1"))
		assert batch_size == 1 or report["frames"] % batch_size, \
			"마지막 불완전 배치를 검증하지 못하는 합성 영상이다"
		assert report.get("batch_size") == batch_size, "적용된 배치 크기가 리포트에 없다"
		expected_batches = len({index // batch_size for index in range(0, report["frames"], INFER_STRIDE)})
		assert report.get("inference_batches") == expected_batches, \
			"마지막 불완전 배치를 포함한 추론 호출 수가 다르다"
		out_cap = cv2.VideoCapture(str(dst))
		assert int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT)) == report["frames"], "출력 프레임 수가 입력과 다르다"
		out_cap.release()
		assert dst.exists() and dst.stat().st_size > 0, "출력 영상이 비었다"
		# MSG-367: 재생본에 mp4v 경유 흔적이 없어야 하고, 무음 입력은 무음으로 완주해야 한다 (-map 1:a?)
		assert ffprobe_codec(dst) == "h264", f"재생본 코덱이 h264가 아니다 (MSG-367): {ffprobe_codec(dst)}"
		assert ffprobe_codec(dst, "a:0") == "", "무음 입력인데 출력에 오디오 스트림이 생겼다 (MSG-367)"
		assert len(report["highlights"]) <= 3, "하이라이트는 최대 3구간 (MSG-141)"
		assert report["highlights"], "5초 이상인데 하이라이트 0개 (MSG-159 폴백 미작동)"
		# MSG-353: 구간 품질 — 각 5초 이상, 시작점 간격 5초 이상 (BE docs/prd/MSG-351-prd.md FR-3)
		for start, end in report["highlights"]:
			assert end - start >= HIGHLIGHT_MIN_SEC - 0.01, f"5초 미만 구간 (MSG-353): {report['highlights']}"
		starts = [start for start, _ in report["highlights"]]
		assert all(abs(a - b) >= HIGHLIGHT_MIN_GAP_SEC - 0.01
			for i, a in enumerate(starts) for b in starts[:i]), \
			f"시작점 간격 5초 미만 (MSG-353): {report['highlights']}"
		assert detect_highlights(src, 4.0, Stage()) == [], "5초 미만은 빈 배열이어야 한다 (MSG-353)"
		# 12초 영상은 시작점 간격 5초 제약상 최대 2구간이다 — 구 균등 3분할이 4초짜리 3개를
		# 만들던 회귀를 여기서 잡는다 (docs/MSG-353.md D-3)
		twelve = Path(tmp) / "twelve.mp4"
		make_smoke_video(twelve, seconds=12)
		twelve_highlights = detect_highlights(twelve, 12.0, Stage())
		assert 1 <= len(twelve_highlights) <= 2, f"12초 영상 구간 수 위반 (MSG-353): {twelve_highlights}"
		for start, end in twelve_highlights:
			assert end - start >= HIGHLIGHT_MIN_SEC - 0.01 and 0 <= start <= 12 - HIGHLIGHT_MIN_SEC + 0.01, \
				f"12초 영상 구간 경계 위반 (MSG-353): {twelve_highlights}"
		assert set(report["stages"]) >= {"decode", "infer_face", "encode"}, "단계 계측 누락"
		assert pad_boxes([(10, 10, 20, 20)], 100, 100, pad=0.1) == [(9, 9, 21, 21)], "박스 확장 오계산"
		assert pad_boxes([(0, 0, 100, 100)], 100, 100, pad=0.2) == [(0, 0, 100, 100)], "확장이 프레임을 벗어남"

		# MSG-339: 검출 0건인 합성 영상은 결과를 다른 프레임에 적용해도 잡지 못한다.
		# 7장에 식별자를 심어 배치 4의 마지막 3장과 두 모델 결과가
		# 입력 순서대로 소비되는지 확인한다.
		palette = [32, 64, 96, 128, 160, 192, 224]
		order_src, order_dst = Path(tmp) / "order.mp4", Path(tmp) / "order-out.mp4"
		order_writer = cv2.VideoWriter(str(order_src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
		for value in palette:
			order_writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
		order_writer.release()

		class ProbeResult:
			def __init__(self, marker, y):
				self.boxes = type("ProbeBoxes", (), {
					"xyxy": np.array([[marker, y, marker + 1, y + 1]], dtype=np.float32),
				})()

		class ProbeModel:
			def __init__(self, y):
				self.y = y
				self.batch_lengths = []

			def predict(self, sources, **_kwargs):
				assert isinstance(sources, list), "Ultralytics에 프레임 목록이 아니라 낱장을 넘겼다"
				self.batch_lengths.append(len(sources))
				return [ProbeResult(min(range(7), key=lambda i: abs(float(frame.mean()) - palette[i])) + 1, self.y)
					for frame in sources]

		face_probe, plate_probe = ProbeModel(0), ProbeModel(2)
		applied = []

		def record_blur(frame, boxes):
			frame_marker = min(range(7), key=lambda i: abs(float(frame.mean()) - palette[i])) + 1
			applied.append((frame_marker, boxes[0][0], boxes[1][0]))
			return frame

		real_load_models, real_blur_boxes = load_models, blur_boxes
		globals()["load_models"] = lambda _device: (face_probe, plate_probe)
		globals()["blur_boxes"] = record_blur
		try:
			order_report = run(order_src, "cpu", order_dst)
		finally:
			globals()["load_models"], globals()["blur_boxes"] = real_load_models, real_blur_boxes
		expected_markers = [(i // INFER_STRIDE) * INFER_STRIDE + 1 for i in range(7)]
		assert applied == [(i + 1, marker, marker) for i, marker in enumerate(expected_markers)], \
			f"배치 결과와 입력 프레임 순서가 어긋났다: {applied}"
		expected_sizes = [sum(i % INFER_STRIDE == 0 for i in range(start, min(start + batch_size, 7)))
			for start in range(0, 7, batch_size)]
		expected_sizes = [size for size in expected_sizes if size]
		assert face_probe.batch_lengths == expected_sizes == plate_probe.batch_lengths, \
			f"모델별 배치 구성이 다르다: face={face_probe.batch_lengths}, plate={plate_probe.batch_lengths}"
		assert order_report["frames"] == 7, "마지막 불완전 배치의 프레임이 출력되지 않았다"

		# MSG-367: 오디오 있는 입력의 왕복 — 재생본에 원본 오디오가 실려야 하고(-map 1:a? + aac),
		# 오디오(3초)가 영상(6초)보다 짧아도 영상이 절단되면 안 된다 — -shortest 단독이던 시절
		# 오디오 EOF에 exit 0으로 절단되던 P1 회귀를 -af apad로 막았다.
		# 해상도를 홀수(639x361)로 잡아 pad 방어(라운드3)도 같은 실경로에서 검증한다 —
		# 출력은 640x362로 패딩되고 프레임 수는 보존돼야 한다
		tone_src, tone_dst = Path(tmp) / "tone.mp4", Path(tmp) / "tone-out.mp4"
		make_smoke_video(tone_src, seconds=6, size=(639, 361), audio_sec=3)
		tone_report = run(tone_src, "cpu", tone_dst)
		assert tone_report["frames"] > 0 and ffprobe_codec(tone_dst) == "h264", \
			f"오디오 입력 재생본이 h264가 아니다 (MSG-367): {ffprobe_codec(tone_dst)}"
		assert ffprobe_codec(tone_dst, "a:0") == "aac", \
			f"오디오가 재생본에 없다 (MSG-367): {ffprobe_codec(tone_dst, 'a:0')!r}"
		tone_cap = cv2.VideoCapture(str(tone_dst))
		assert int(tone_cap.get(cv2.CAP_PROP_FRAME_COUNT)) == tone_report["frames"], \
			"짧은 오디오 입력에서 영상이 절단됐다 (MSG-367 -shortest+apad 회귀)"
		tone_w = int(tone_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		tone_h = int(tone_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		tone_cap.release()
		assert (tone_w, tone_h) == (640, 362), \
			f"홀수 해상도가 짝수로 패딩되지 않았다 (MSG-367 라운드3): {tone_w}x{tone_h}"

		# MSG-367 D-4: ffmpeg가 죽으면(못 여는 출력 경로) cv2.VideoWriter처럼 조용히 빈 파일을
		# 남기지 않고 예외로 드러나야 한다 — 워커가 이 예외를 잡아 잡을 FAILED로 만든다
		globals()["load_models"] = lambda _device: (face_probe, plate_probe)
		try:
			try:
				run(order_src, "cpu", Path(tmp) / "no-such-dir" / "fail.mp4")
				raise AssertionError("출력 경로를 못 여는데 예외가 안 났다 (MSG-367 D-4)")
			except RuntimeError as e:
				assert "ffmpeg" in str(e), f"실패 예외에 ffmpeg 진단이 없다 (MSG-367 D-4): {e}"
		finally:
			globals()["load_models"] = real_load_models

		# MSG-367 라운드2: 추론·블러 예외 경로(loop_ok=False, kill 선행) — finally가 인코더를
		# 회수하면서 원예외를 치환·유실하지 않고 그대로 전파해야 한다. 위 D-4 케이스는
		# BrokenPipe→정상 루프 이탈 경로라 이 분기를 타지 않는다
		def exploding_blur(frame, boxes):
			raise ValueError("주입한 블러 실패")

		globals()["load_models"] = lambda _device: (face_probe, plate_probe)
		globals()["blur_boxes"] = exploding_blur
		try:
			try:
				run(order_src, "cpu", Path(tmp) / "boom.mp4")
				raise AssertionError("주입한 예외가 전파되지 않았다 (MSG-367 라운드2)")
			except ValueError as e:
				assert "주입한 블러 실패" in str(e), f"원예외가 치환됐다 (MSG-367 라운드2): {e}"
		finally:
			globals()["load_models"], globals()["blur_boxes"] = real_load_models, real_blur_boxes

		# MSG-280: 납작한 박스가 실제로 뭉개지는지. 합성 영상엔 검출 대상이 없어 blur_boxes가
		# 안 타므로 직접 태운다. 30×120에 15px 폭 블록 = 번호판 글자 굵기 근사.
		# 실측 잔여 std: 짧은 변 기준 0.91배(글자 읽힘) / 긴 변 기준 0.65배 — 사이인 0.8을 문턱으로.
		strip = np.zeros((30, 120, 3), dtype=np.uint8)
		for i in range(0, 120, 30):
			strip[:, i:i + 15] = 255
		assert blur_boxes(strip.copy(), [(0, 0, 120, 30)]).std() < 0.8 * strip.std(), \
			"납작한 박스에 블러가 약하다 — 커널이 짧은 변 기준인지 확인 (MSG-280)"

		# MSG-284 프리체크. 단색은 대비가 0이라 밝기와 무관하게 탈락 방향이어야 한다.
		black = np.zeros((120, 160, 3), dtype=np.uint8)
		white = np.full((120, 160, 3), 255, dtype=np.uint8)
		assert frame_metrics(black)["std"] < PRECHECK_STD_THRESHOLD, "검은 단색이 탈락 방향이 아니다"
		assert frame_metrics(white)["std"] < PRECHECK_STD_THRESHOLD, "흰 단색이 탈락 방향이 아니다"
		# 센서 노이즈가 낀 암흑 프레임(리포트의 FAIL 샘플 구성)도 탈락해야 한다 —
		# 노이즈는 lap_var를 키우지만 std는 못 키운다는 게 std를 택한 이유다
		noisy = np.random.default_rng(0).integers(0, 12, (120, 160, 3), dtype=np.uint8)
		assert frame_metrics(noisy)["std"] < PRECHECK_STD_THRESHOLD, "노이즈 낀 암흑이 통과 방향이다"

		ok = precheck(src)
		assert ok["passed"] and ok["reason"] is None, f"컬러바 영상이 탈락했다(오탐): {ok}"

		dark = Path(tmp) / "dark.mp4"
		make_dark_video(dark, dark_sec=6)
		bad = precheck(dark)
		assert not bad["passed"], f"암흑 영상이 통과했다(미탐): {bad}"
		assert bad["reason"].startswith("too_dark:") and "std" in bad["reason"], \
			f"reason 형식이 계약과 다르다: {bad['reason']}"

		# 중앙값 규칙의 오탐 방지 — 촬영 시작 직후에만 가려진 영상은 통과해야 한다
		head_dark = Path(tmp) / "head-dark.mp4"
		make_dark_video(head_dark, dark_sec=1, bright_sec=5)
		assert precheck(head_dark)["passed"], "앞부분만 어두운 영상이 탈락했다 — 중앙값 규칙 미작동"

		# 프레임 확보를 시킹에서 순차 grab으로 바꿨다(MSG-284 후속). **뽑는 인덱스가 시킹판과
		# 같아야** results/MSG-284-precheck.json·MSG-281-grid.json 같은 기존 실측 근거가 유효하다.
		# n이 총 프레임보다 크면 같은 인덱스가 연속으로 나온다 — 순차 재생은 되감을 수 없는 경로다.
		cap = cv2.VideoCapture(str(src))
		for n in (7, report["frames"] * 2):
			picked = sample_frames(src, n)
			assert [i for i, _ in picked] == [int(i * report["frames"] / n) for i in range(n)], \
				f"n={n}: 균등 인덱스가 어긋났다 — 기존 실측과 다른 프레임을 뽑는다 (MSG-284)"
			# 인덱스는 append 때 붙이는 라벨이라 인덱스만 보면 "라벨은 맞고 프레임이 한 장 밀린"
			# off-by-one이 통과한다. 이 변경이 지켜야 하는 성질은 같은 픽셀이라 한 장은 시킹으로 직접 대조한다.
			want, frame = picked[len(picked) // 2]
			cap.set(cv2.CAP_PROP_POS_FRAMES, want)
			read_ok, expected = cap.read()  # 위쪽 precheck 결과 ok를 가리지 않도록 이름을 따로 쓴다
			assert read_ok and np.array_equal(frame, expected), \
				f"n={n}: 인덱스 {want}의 픽셀이 시킹판과 다르다 (MSG-284)"
		cap.release()

		# 프레임 0장은 기존 실패 경로(SystemExit)를 그대로 탄다 (MSG-284 FR-8)
		try:
			video_metrics(src, 0)
			raise AssertionError("프레임 0장인데 SystemExit이 안 났다")
		except SystemExit:
			pass

		# 부분 디코딩(손상 입력)이 "너무 어두움" 탈락으로 오진되지 않고 실패 경로로 가는지 (FR-8).
		# 실제 부분 디코딩을 만들기 어려워 sample_frames를 잠깐 갈아끼운다 — 가드는 precheck 안에 있다
		real_sample = sample_frames
		globals()["sample_frames"] = lambda path, n: real_sample(path, 2)
		try:
			precheck(src, 10)
			raise AssertionError("표본이 절반 미만인데 SystemExit이 안 났다")
		except SystemExit:
			pass
		finally:
			globals()["sample_frames"] = real_sample

		print(json.dumps(report, indent=2, ensure_ascii=False))
		print(f"precheck: 컬러바={ok['metrics']['std']} / 암흑={bad['metrics']['std']} ({bad['reason']})")
		print("\nsmoke OK")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("video", nargs="?", help="측정할 입력 영상")
	ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
	ap.add_argument("--out", default="out.mp4", help="블러 처리본 저장 경로")
	ap.add_argument("--smoke", action="store_true", help="합성 영상으로 파이프라인 검증")
	args = ap.parse_args()

	if args.smoke or not args.video:
		smoke()
		return
	print(json.dumps(run(Path(args.video), args.device, Path(args.out)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
