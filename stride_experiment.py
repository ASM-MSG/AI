"""MSG-207 — 프레임 스킵(stride) 커버리지 시뮬레이션.

전 프레임 추론 박스를 수집한 뒤(pseudo-GT), stride·pad 정책을 오프라인으로 시뮬레이션한다:
스킵 프레임의 GT 박스가 직전 앵커 프레임의 확장 박스로 얼마나 가려지는지(coverage).
영상당 전 프레임 추론 1회로 모든 정책 조합을 비교한다 (MSG-158 pseudo-GT 방법론 계열).

    python stride_experiment.py samples/crowd.mp4 samples/plates.mp4
    python stride_experiment.py --smoke

GT는 배포 백엔드와 같은 걸로 수집할 것 — AI_BACKEND=openvino 로 실행하면 수집도 openvino.
"""

import argparse
import json
from pathlib import Path

import bench

STRIDES = [2, 3, 4]
PADS = [0.0, 0.1, 0.2]
COVERED = 0.8  # GT 박스 면적의 80% 이상 가려지면 '커버' — 블러 목적이라 완전 일치보다 가림 비율이 기준


def collect(video):
	"""전 프레임 추론 → 프레임별 {"face": [...], "plate": [...]} 박스."""
	import cv2

	face, plate = bench.load_models("cpu")
	cap = cv2.VideoCapture(str(video))
	if not cap.isOpened():
		raise SystemExit(f"영상을 열 수 없음: {video}")
	w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out = []
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		fr = face.predict(frame, conf=bench.FACE_CONF, verbose=False)[0]
		pr = plate.predict(frame, conf=bench.PLATE_CONF, verbose=False)[0]
		out.append({"face": bench.to_boxes(fr, w, h), "plate": bench.to_boxes(pr, w, h)})
	cap.release()
	return out, w, h


def cover_ratio(gt, reused):
	"""GT 박스가 재사용 박스 하나로 가려지는 최대 비율 (교집합/GT 면적)."""
	gx1, gy1, gx2, gy2 = gt
	area = max(1, (gx2 - gx1) * (gy2 - gy1))
	best = 0.0
	for x1, y1, x2, y2 in reused:
		iw = min(gx2, x2) - max(gx1, x1)
		ih = min(gy2, y2) - max(gy1, y1)
		if iw > 0 and ih > 0:
			best = max(best, iw * ih / area)
	return best


def simulate(per_frame, stride, pad, w, h):
	"""스킵 프레임만 측정한다 — 앵커 프레임은 실제 추론이라 커버리지 1.0."""
	stats = {c: {"n": 0, "covered": 0, "ratio_sum": 0.0} for c in ("face", "plate")}
	anchor = {"face": [], "plate": []}
	for i, boxes in enumerate(per_frame):
		if i % stride == 0:
			anchor = {c: bench.pad_boxes(boxes[c], w, h, pad=pad) for c in boxes}
			continue
		for c in ("face", "plate"):
			for gt in boxes[c]:
				r = cover_ratio(gt, anchor[c])
				s = stats[c]
				s["n"] += 1
				s["ratio_sum"] += r
				s["covered"] += r >= COVERED
	return {
		c: {
			"gt_boxes": s["n"],
			"mean_cover": round(s["ratio_sum"] / s["n"], 4) if s["n"] else None,
			"covered_rate": round(s["covered"] / s["n"], 4) if s["n"] else None,
		}
		for c, s in stats.items()
	}


def run_grid(videos, out_path):
	grid = {}
	for video in videos:
		per_frame, w, h = collect(video)
		rows = {}
		for stride in STRIDES:
			for pad in PADS:
				rows[f"s{stride}-p{pad}"] = simulate(per_frame, stride, pad, w, h)
		grid[Path(video).name] = {"frames": len(per_frame), "grid": rows}
		print(f"\n== {Path(video).name} ({len(per_frame)} frames)")
		print(f"{'설정':>10} | {'face cover/rate':>18} | {'plate cover/rate':>18}")
		for key, res in rows.items():
			f, p = res["face"], res["plate"]
			print(f"{key:>10} | {str(f['mean_cover']):>8}/{str(f['covered_rate']):>9} | "
			      f"{str(p['mean_cover']):>8}/{str(p['covered_rate']):>9}")
	Path(out_path).write_text(json.dumps(grid, indent=2, ensure_ascii=False))
	print(f"\n저장: {out_path}")


def smoke():
	# 프레임당 2px씩 움직이는 40px 얼굴 박스 — pad가 이동을 흡수하는지 검증
	moving = [{"face": [(100 + 2 * i, 100, 140 + 2 * i, 160)], "plate": []} for i in range(30)]
	static = [{"face": [(100, 100, 140, 160)], "plate": []} for _ in range(30)]

	assert simulate(static, 3, 0.0, 640, 480)["face"]["covered_rate"] == 1.0, "정지 박스는 pad 0에서도 완전 커버여야 한다"
	no_pad = simulate(moving, 3, 0.0, 640, 480)["face"]["mean_cover"]
	padded = simulate(moving, 3, 0.2, 640, 480)["face"]["mean_cover"]
	assert padded > no_pad, "pad가 이동 박스 커버리지를 올려야 한다"
	assert simulate(moving, 3, 0.2, 640, 480)["face"]["covered_rate"] == 1.0, "2px/f 이동은 pad 0.2로 커버돼야 한다"
	assert cover_ratio((0, 0, 10, 10), [(20, 20, 30, 30)]) == 0.0, "교집합 없음은 0"
	print("smoke OK")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("videos", nargs="*", help="측정할 영상들")
	ap.add_argument("--out", default="results/MSG-207-stride.json")
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()

	if args.smoke or not args.videos:
		smoke()
		return
	run_grid(args.videos, args.out)


if __name__ == "__main__":
	main()
