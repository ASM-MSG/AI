"""MSG-142 — 블러·하이라이트 파이프라인 처리 시간·리소스 측정.

병목이 AI인지 인코딩인지 가르는 게 목적이라 단계별로 시간을 따로 누적한다.
정확도 튜닝은 범위 밖 (MSG-142 티켓 명시).

    python bench.py --smoke                        # 합성 영상으로 파이프라인 검증
    python bench.py sample.mp4 --device cpu
    python bench.py sample.mp4 --device mps        # Apple Silicon
    python bench.py sample.mp4 --device cuda       # AWS GPU 인스턴스

MSG-207 실험 노브 (환경변수) — **둘 다 실측 결과 미채택, 기본값이 정본** (results/MSG-207-report.md):
    AI_BACKEND=openvino       # +5%뿐이라 미채택. 쓰려면 openvino 별도 설치 필요
    AI_INFER_STRIDE=3         # 도로 씬 번호판 커버 46%로 붕괴 — 프라이버시 사고라 미채택
    AI_BOX_PAD=0.10           # 스킵 프레임 재사용 박스 확장 비율 (stride>1일 때만 의미)
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

# MSG-207: 추론 백엔드·프레임 스트라이드. 실측 결과 두 노브 모두 미채택 — 기본값이 정본.
BACKEND = os.environ.get("AI_BACKEND", "torch")             # torch | openvino
INFER_STRIDE = int(os.environ.get("AI_INFER_STRIDE", "1"))  # k프레임마다 1회 추론
BOX_PAD = float(os.environ.get("AI_BOX_PAD", "0.10"))       # 재사용 박스 확장 비율


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
	"""PySceneDetect 장면 전환 기준 상위 3구간. 5초 미만이면 건너뛴다 (MSG-141)."""
	from scenedetect import ContentDetector, detect

	with stage.track("highlight"):
		scenes = detect(str(path), ContentDetector())
	if not scenes:
		# MSG-159: 한 자리 촬영엔 장면 전환이 없어 0개가 나온다 → 균등 3분할 폴백.
		# ponytail: 균등 분할 — 추천 품질 불만이 실측되면 움직임량 랭킹으로 승격
		third = duration / 3
		return [[round(i * third, 2), round((i + 1) * third, 2)] for i in range(3)]
	ranked = sorted(scenes, key=lambda s: (s[1] - s[0]).get_seconds(), reverse=True)
	return [[round(s.get_seconds(), 2), round(e.get_seconds(), 2)] for s, e in ranked[:3]]


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


def run(path, device, out_path):
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
	writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

	frames = face_hits = plate_hits = 0
	reused = []
	while True:
		with stage.track("decode"):
			ok, frame = cap.read()
		if not ok:
			break
		frames += 1

		if (frames - 1) % INFER_STRIDE == 0:
			with stage.track("infer_face"):
				fr = face.predict(frame, conf=FACE_CONF, verbose=False)[0]
			with stage.track("infer_plate"):
				pr = plate.predict(frame, conf=PLATE_CONF, verbose=False)[0]

			fb, pb = to_boxes(fr, width, height), to_boxes(pr, width, height)
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
			writer.write(frame)

	cap.release()
	writer.release()

	duration = frames / fps if fps else 0
	highlights = detect_highlights(path, duration, stage) if duration >= 5 else []
	wall = time.perf_counter() - wall_start

	infer = stage.totals.get("infer_face", 0) + stage.totals.get("infer_plate", 0)
	return {
		"input": str(path),
		"device": device,
		"backend": BACKEND,
		"infer_stride": INFER_STRIDE,
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


def make_smoke_video(path, seconds=6, fps=30, size=(640, 480)):
	"""검출 결과가 0건이어도 파이프라인은 완주해야 한다 (MSG-140 완료 조건)."""
	subprocess.run(
		["ffmpeg", "-y", "-f", "lavfi", "-i",
		 f"testsrc=duration={seconds}:size={size[0]}x{size[1]}:rate={fps}",
		 "-pix_fmt", "yuv420p", str(path)],
		check=True, capture_output=True,
	)


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
		make_smoke_video(src)
		report = run(src, "cpu", dst)

		assert report["frames"] > 0, "프레임을 하나도 읽지 못했다"
		assert dst.exists() and dst.stat().st_size > 0, "출력 영상이 비었다"
		assert len(report["highlights"]) <= 3, "하이라이트는 최대 3구간 (MSG-141)"
		assert report["highlights"], "5초 이상인데 하이라이트 0개 (MSG-159 폴백 미작동)"
		assert set(report["stages"]) >= {"decode", "infer_face", "encode"}, "단계 계측 누락"
		assert pad_boxes([(10, 10, 20, 20)], 100, 100, pad=0.1) == [(9, 9, 21, 21)], "박스 확장 오계산"
		assert pad_boxes([(0, 0, 100, 100)], 100, 100, pad=0.2) == [(0, 0, 100, 100)], "확장이 프레임을 벗어남"

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
