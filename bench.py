"""MSG-142 — 블러·하이라이트 파이프라인 처리 시간·리소스 측정.

병목이 AI인지 인코딩인지 가르는 게 목적이라 단계별로 시간을 따로 누적한다.
정확도 튜닝은 범위 밖 (MSG-142 티켓 명시).

    python bench.py --smoke                        # 합성 영상으로 파이프라인 검증
    python bench.py sample.mp4 --device cpu
    python bench.py sample.mp4 --device mps        # Apple Silicon
    python bench.py sample.mp4 --device cuda       # AWS GPU 인스턴스
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

# (repo, 가중치 파일)을 못박는다. 번호판 레포는 n/s/m/l/x 5종을 같은 길이 이름으로
# 올려둬서 파일명을 추측하면 조용히 large를 집는다 — nano를 명시할 것.
FACE_MODEL = ("AdamCodd/YOLOv11n-face-detection", "model.pt")
PLATE_MODEL = ("morsetechlab/yolov11-license-plate-detection", "license-plate-finetune-v1n.pt")

# 블러는 과하게 걸려도 손해가 작지만 미탐지는 프라이버시 사고다 → 기본값을 낮게 잡는다.
# 얼굴 0.05는 MSG-158 실험 결과 (results/MSG-158-report.md) — recall 0.72→0.98, 시간 불변.
# 번호판은 미실험이라 0.25 유지.
FACE_CONF = 0.05
PLATE_CONF = 0.25


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


def load_models(device):
	from huggingface_hub import hf_hub_download
	from ultralytics import YOLO

	face = YOLO(hf_hub_download(*FACE_MODEL))
	plate = YOLO(hf_hub_download(*PLATE_MODEL))
	face.to(device)
	plate.to(device)
	return face, plate


def blur_boxes(frame, boxes):
	"""검출 영역만 잘라 가우시안 블러 후 되붙인다."""
	for x1, y1, x2, y2 in boxes:
		roi = frame[y1:y2, x1:x2]
		if roi.size == 0:
			continue
		# 커널을 ROI 크기에 비례시켜야 작은 얼굴도 실제로 뭉개진다.
		k = max(3, (min(roi.shape[:2]) // 4) | 1)
		frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
	return frame


def to_boxes(result, width, height):
	out = []
	for box in result.boxes.xyxy.tolist():
		x1, y1, x2, y2 = (int(v) for v in box)
		out.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
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


def run(path, device, out_path):
	stage = Stage()
	wall_start = time.perf_counter()

	with stage.track("model_load"):
		face, plate = load_models(device)

	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise SystemExit(f"영상을 열 수 없음: {path}")

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

	frames = face_hits = plate_hits = 0
	while True:
		with stage.track("decode"):
			ok, frame = cap.read()
		if not ok:
			break
		frames += 1

		with stage.track("infer_face"):
			fr = face.predict(frame, conf=FACE_CONF, verbose=False)[0]
		with stage.track("infer_plate"):
			pr = plate.predict(frame, conf=PLATE_CONF, verbose=False)[0]

		fb, pb = to_boxes(fr, width, height), to_boxes(pr, width, height)
		face_hits += len(fb)
		plate_hits += len(pb)

		with stage.track("mask"):
			frame = blur_boxes(frame, fb + pb)
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
		print(json.dumps(report, indent=2, ensure_ascii=False))
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
