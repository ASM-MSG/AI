"""MSG-281 — 추론 해상도 하향 실험: imgsz 640 → 480 → 320.

건당 2.6분 중 추론이 80%다(MSG-142). 연산량은 imgsz²에 비례하니 하향이 남은 소프트웨어
레버 중 가장 큰데 한 번도 재본 적이 없다 — MSG-158은 상향(960·1280)만 봤고 역효과였다.

현재 설정(640)의 검출을 pseudo-GT로 놓고 하향판이 그중 몇 %를 따라잡는지 잰다.
640이 원래 놓치던 것은 보이지 않는다 — "지금 대비 얼마나 잃는가"를 재는 상대 지표다.

**작은 객체가 해상도 하향에 먼저 무너지므로 전체 recall만으로는 판단할 수 없다.**
COCO 관행대로 박스 면적을 small/medium/large로 갈라 함께 낸다.

    python imgsz_experiment.py --smoke                     # 합성 영상으로 완주 검증
    python imgsz_experiment.py --device mps                # 기본 샘플 4종
    python imgsz_experiment.py samples/kr-plates.mp4 --frames 40

정확도(recall)는 디바이스 무관이라 macOS로도 유효하지만, **ms/frame은 EC2에서만 유효하다**
(CLAUDE.md "알아둘 함정" — Apple Silicon PyTorch가 스레드 설정을 무시한다).

출력:
    results/MSG-281-grid.json                              # 수치 (커밋 가능)
    samples/msg281/<target>/*.jpg                          # 해상도별 박스 겹쳐 그린 육안 검증용 (커밋 금지)
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2

from bench import FACE_CONF, FACE_MODEL, PLATE_CONF, PLATE_MODEL, make_smoke_video, to_boxes
from face_experiment import IOU_MATCH, iou, sample_frames

# 비교할 해상도. **첫 값이 현재 파이프라인 설정이자 pseudo-GT 기준**이다 — 순서를 바꾸지 말 것.
IMGSZ_GRID = [640, 480, 320]

# 대상별 (모델, conf). 번호판은 납작하고 원거리라 얼굴보다 해상도에 민감할 것으로 예상돼
# 반드시 따로 잰다 — 합쳐서 평균 내면 번호판 붕괴가 얼굴 recall에 묻힌다.
TARGETS = {
	"face": (FACE_MODEL, FACE_CONF),
	"plate": (PLATE_MODEL, PLATE_CONF),
}

# COCO 관행의 면적 경계(32²·96²). 작은 객체가 먼저 무너지므로 구간을 갈라 본다.
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96
BUCKETS = ["small", "medium", "large"]

# 해상도별 박스 색 (BGR) — 한 프레임에 겹쳐 그려 무엇이 사라지는지 눈으로 본다.
BUCKET_COLORS = {640: (0, 255, 0), 480: (0, 255, 255), 320: (0, 0, 255)}

ANNOTATE_EVERY = 10


def size_bucket(box):
	x1, y1, x2, y2 = box
	area = (x2 - x1) * (y2 - y1)
	if area < SMALL_AREA:
		return "small"
	return "medium" if area < MEDIUM_AREA else "large"


def detect_all(model, conf, imgsz, frames_by_video, device):
	"""해상도 하나로 전 프레임 검출. {(video, idx): [box]} 와 프레임당 추론 ms를 돌려준다."""
	boxes_out = {}
	infer_sec = 0.0
	n = 0
	for video, frames in frames_by_video.items():
		model.predict(frames[0][1], conf=conf, imgsz=imgsz, verbose=False)  # warmup
		for idx, frame in frames:
			h, w = frame.shape[:2]
			t0 = time.perf_counter()
			result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
			infer_sec += time.perf_counter() - t0
			n += 1
			boxes_out[(video, idx)] = to_boxes(result, w, h)
	return boxes_out, (1000 * infer_sec / n if n else 0.0)


def recall_by_size(gt_boxes, cand_boxes):
	"""GT 박스를 크기 구간별로 나눠 매칭률을 낸다. GT가 0개인 구간은 None."""
	matched = {b: 0 for b in BUCKETS}
	total = {b: 0 for b in BUCKETS}
	for key, gts in gt_boxes.items():
		cand = cand_boxes.get(key, [])
		for g in gts:
			bucket = size_bucket(g)
			total[bucket] += 1
			if any(iou(g, c) >= IOU_MATCH for c in cand):
				matched[bucket] += 1
	out = {b: (round(matched[b] / total[b], 3) if total[b] else None) for b in BUCKETS}
	gt_total = sum(total.values())
	out["overall"] = round(sum(matched.values()) / gt_total, 3) if gt_total else None
	out["gt_counts"] = dict(total)
	return out


def annotate(frames_by_video, boxes_by_imgsz, out_dir):
	"""해상도별 박스를 한 프레임에 겹쳐 그린다 — 640에만 있고 320에 없는 박스가 미탐지다."""
	out_dir.mkdir(parents=True, exist_ok=True)
	for video, frames in frames_by_video.items():
		for i, (idx, frame) in enumerate(frames):
			if i % ANNOTATE_EVERY:
				continue
			vis = frame.copy()
			for order, imgsz in enumerate(IMGSZ_GRID):
				color = BUCKET_COLORS.get(imgsz, (255, 255, 255))
				pad = order * 3  # 겹친 박스가 서로 가리지 않게 조금씩 키워 그린다
				for x1, y1, x2, y2 in boxes_by_imgsz[imgsz].get((video, idx), []):
					cv2.rectangle(vis, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), color, 2)
				cv2.putText(vis, f"imgsz {imgsz}", (10, 40 + 35 * order),
					cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
			cv2.imwrite(str(out_dir / f"{Path(video).stem}-{idx:05d}.jpg"), vis)


def run_grid(videos, n_frames, device, annotate_root):
	from huggingface_hub import hf_hub_download
	from ultralytics import YOLO

	frames_by_video = {str(v): sample_frames(v, n_frames) for v in videos}
	gt_imgsz = IMGSZ_GRID[0]
	results = {}

	for target, (model_spec, conf) in TARGETS.items():
		model = YOLO(hf_hub_download(*model_spec))
		model.to(device)

		boxes_by_imgsz = {}
		ms_by_imgsz = {}
		for imgsz in IMGSZ_GRID:
			print(f"[{target}] imgsz={imgsz} ...", flush=True)
			boxes_by_imgsz[imgsz], ms_by_imgsz[imgsz] = detect_all(
				model, conf, imgsz, frames_by_video, device)

		rows = []
		for imgsz in IMGSZ_GRID:
			rec = recall_by_size(boxes_by_imgsz[gt_imgsz], boxes_by_imgsz[imgsz])
			rows.append({
				"imgsz": imgsz,
				"detections": sum(len(b) for b in boxes_by_imgsz[imgsz].values()),
				"recall_vs_640": rec,
				"ms_per_frame": round(ms_by_imgsz[imgsz], 1),
				"speedup": round(ms_by_imgsz[gt_imgsz] / ms_by_imgsz[imgsz], 2) if ms_by_imgsz[imgsz] else None,
			})
		results[target] = rows

		if annotate_root:
			annotate(frames_by_video, boxes_by_imgsz, annotate_root / target)
		del model

	return {
		"videos": [str(v) for v in videos],
		"frames_per_video": n_frames,
		"device": device,
		"gt_imgsz": gt_imgsz,
		"note": "recall은 imgsz 640 검출을 기준으로 한 상대 지표(640이 놓친 것은 안 보인다). "
			"ms/frame은 EC2에서만 유효. 최종 판단은 주석 프레임 육안 검증.",
		"results": results,
	}


def print_table(report):
	for target, rows in report["results"].items():
		print(f"\n== {target} ==")
		head = f"{'imgsz':>6} {'det':>6} {'overall':>8} {'small':>7} {'medium':>7} {'large':>7} {'ms/frame':>9} {'배수':>6}"
		print(head + "\n" + "-" * len(head))
		for r in rows:
			rec = r["recall_vs_640"]
			f = lambda v: f"{v:.3f}" if isinstance(v, float) else "-"
			sp = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
			print(f"{r['imgsz']:>6} {r['detections']:>6} {f(rec['overall']):>8} {f(rec['small']):>7} "
				f"{f(rec['medium']):>7} {f(rec['large']):>7} {r['ms_per_frame']:>9.1f} {sp:>6}")
		gt = rows[0]["recall_vs_640"]["gt_counts"]
		print(f"       GT 박스 수: small={gt['small']} medium={gt['medium']} large={gt['large']}")


def smoke():
	"""합성 영상(검출 0건)으로 지표 뼈대를 완주시킨다. GT가 없을 때 죽지 않는지가 요점."""
	assert size_bucket((0, 0, 10, 10)) == "small", "31px 미만은 small"
	assert size_bucket((0, 0, 50, 50)) == "medium", "32~96px는 medium"
	assert size_bucket((0, 0, 200, 200)) == "large", "96px 초과는 large"
	# 납작한 번호판(111x32=3552px²)은 medium — 짧은 변이 아니라 면적으로 가른다
	assert size_bucket((0, 0, 111, 32)) == "medium", "번호판 크기 구간 오판"

	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "in.mp4"
		make_smoke_video(src)
		report = run_grid([src], 3, "cpu", None)

		for target, rows in report["results"].items():
			assert [r["imgsz"] for r in rows] == IMGSZ_GRID, f"{target}: 해상도 누락"
			assert all(r["ms_per_frame"] > 0 for r in rows), f"{target}: 추론 시간 계측 실패"
			# 합성 영상엔 얼굴도 번호판도 없다 → GT 0건, recall은 None이어야 하고 죽지 않아야 한다
			assert rows[0]["recall_vs_640"]["overall"] is None, f"{target}: 합성 영상에 GT가 있을 리 없다"
		print_table(report)
		print("\nsmoke OK")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("videos", nargs="*",
		default=["samples/crowd.mp4", "samples/plates.mp4", "samples/kr-faces.mp4", "samples/kr-plates.mp4"],
		help="실험 대상 영상 (기본: 군중·도로 + 한국 클립)")
	ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
	ap.add_argument("--frames", type=int, default=30, help="영상당 샘플 프레임 수")
	ap.add_argument("--out", default="results/MSG-281-grid.json")
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()

	if args.smoke:
		smoke()
		return

	report = run_grid([Path(v) for v in args.videos], args.frames, args.device, Path("samples/msg281"))
	Path(args.out).parent.mkdir(exist_ok=True)
	Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
	print_table(report)
	print(f"\n수치: {args.out}\n주석 프레임: samples/msg281/<target>/")


if __name__ == "__main__":
	main()
