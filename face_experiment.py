"""MSG-158 — 얼굴 미탐지 개선: conf 하향 / 모델 크기 / imgsz 상향 비교 실험.

MSG-142 육안 확인에서 군중 정면 얼굴·차량 유리 너머 운전자를 다수 놓쳤다.
세 축을 한 번에 하나씩만 움직여(one-axis-at-a-time) 어느 축이 효과적인지 가른다.

라벨링된 GT가 없으므로 recall은 합의 기반 pseudo-GT로 근사한다:
2개 이상 설정이 동의한(IoU≥0.5) 박스를 정답으로 놓고 설정별 매칭률을 잰다.
절대 정확도가 아니라 설정 간 순위를 보는 지표다. 최종 판단은 주석 프레임 육안 검증.

    python face_experiment.py --smoke              # 합성 영상으로 완주 검증
    python face_experiment.py                      # samples/crowd.mp4 + plates.mp4
    python face_experiment.py --device mps --frames 30

출력:
    results/MSG-158-grid.json                      # 수치 (커밋 가능)
    samples/msg158/<config>/*.jpg                  # 육안 검증용 주석 프레임 (커밋 금지)
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2

from bench import FACE_MODEL, make_smoke_video, to_boxes

# 모델 크기 축은 deepghs(akanametov WIDER FACE 학습) 한 패밀리로 통일한다 —
# AdamCodd n과 학습 데이터가 달라서, deepghs n을 끼워 넣어야 크기 효과가 분리된다.
# AdamCodd에는 n/x만 있고 s/m 레포는 존재하지 않는다 (2026-07-21 확인).
MODELS = {
	"adamcodd-n": FACE_MODEL,
	"deepghs-n": ("deepghs/yolo-face", "yolov11n-face/model.pt"),
	"deepghs-s": ("deepghs/yolo-face", "yolov11s-face/model.pt"),
	"deepghs-m": ("deepghs/yolo-face", "yolov11m-face/model.pt"),
}

# (이름, 모델, conf, imgsz) — 첫 줄이 현재 파이프라인 설정 (bench.py와 동일).
CONFIGS = [
	("baseline", "adamcodd-n", 0.25, 640),
	("conf-0.15", "adamcodd-n", 0.15, 640),
	("conf-0.10", "adamcodd-n", 0.10, 640),
	("conf-0.05", "adamcodd-n", 0.05, 640),
	("model-v11n", "deepghs-n", 0.25, 640),
	("model-v11s", "deepghs-s", 0.25, 640),
	("model-v11m", "deepghs-m", 0.25, 640),
	("imgsz-960", "adamcodd-n", 0.25, 960),
	("imgsz-1280", "adamcodd-n", 0.25, 1280),
]

IOU_MATCH = 0.5
MIN_VOTES = 2  # 합의 GT에 넣으려면 몇 개 설정이 동의해야 하는가
ANNOTATE_EVERY = 5  # 샘플 프레임 중 몇 장에 한 번 주석 이미지를 남길까


def sample_frames(path, n):
	"""영상에서 n프레임을 균등 간격으로 뽑는다."""
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise SystemExit(f"영상을 열 수 없음: {path}")
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	out = []
	for i in range(n):
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n))
		ok, frame = cap.read()
		if ok:
			out.append((int(i * total / n), frame))
	cap.release()
	return out


def iou(a, b):
	ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
	iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
	inter = ix * iy
	union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
	return inter / union if union else 0.0


def consensus(boxes_by_config):
	"""프레임 하나에서 MIN_VOTES개 이상 설정이 동의한 박스만 pseudo-GT로 남긴다.

	ponytail: 첫 등장 박스를 클러스터 대표로 쓰는 greedy 매칭 —
	박스가 프레임당 수십 개 수준이라 헝가리안까지 갈 이유가 없다.
	"""
	clusters = []  # [(대표 박스, 동의한 설정 집합)]
	for name, boxes in boxes_by_config.items():
		for b in boxes:
			for c in clusters:
				if iou(b, c[0]) >= IOU_MATCH:
					c[1].add(name)
					break
			else:
				clusters.append((b, {name}))
	return [b for b, votes in clusters if len(votes) >= MIN_VOTES]


def run_config(name, model_key, conf, imgsz, frames_by_video, device, annotate_dir):
	from huggingface_hub import hf_hub_download
	from ultralytics import YOLO

	model = YOLO(hf_hub_download(*MODELS[model_key]))
	model.to(device)

	boxes_out = {}  # (video, frame_idx) -> [box]
	infer_sec = 0.0
	n_inferred = 0
	for video, frames in frames_by_video.items():
		model.predict(frames[0][1], conf=conf, imgsz=imgsz, verbose=False)  # warmup
		for i, (idx, frame) in enumerate(frames):
			h, w = frame.shape[:2]
			t0 = time.perf_counter()
			result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
			infer_sec += time.perf_counter() - t0
			n_inferred += 1
			boxes = to_boxes(result, w, h)
			boxes_out[(video, idx)] = boxes

			if annotate_dir and i % ANNOTATE_EVERY == 0:
				vis = frame.copy()
				for x1, y1, x2, y2 in boxes:
					cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.putText(vis, f"{name} faces={len(boxes)}", (10, 40),
					cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
				dst = annotate_dir / name
				dst.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(dst / f"{Path(video).stem}-{idx:05d}.jpg"), vis)

	del model
	return boxes_out, (1000 * infer_sec / n_inferred if n_inferred else 0.0)


def run_grid(videos, n_frames, device, annotate_dir, configs=CONFIGS):
	frames_by_video = {str(v): sample_frames(v, n_frames) for v in videos}

	all_boxes = {}  # config -> {(video, idx): [box]}
	ms_per_frame = {}
	for name, model_key, conf, imgsz in configs:
		print(f"[{name}] model={model_key} conf={conf} imgsz={imgsz} ...", flush=True)
		all_boxes[name], ms_per_frame[name] = run_config(
			name, model_key, conf, imgsz, frames_by_video, device, annotate_dir)

	# 프레임별 합의 GT → 설정별 recall
	gt_total = 0
	matched = {name: 0 for name, *_ in configs}
	for key in next(iter(all_boxes.values())):
		gt = consensus({name: boxes[key] for name, boxes in all_boxes.items()})
		gt_total += len(gt)
		for name, boxes in all_boxes.items():
			matched[name] += sum(1 for g in gt if any(iou(g, b) >= IOU_MATCH for b in boxes[key]))

	base_ms = ms_per_frame["baseline"] if "baseline" in ms_per_frame else None
	rows = []
	for name, model_key, conf, imgsz in configs:
		total_faces = sum(len(b) for b in all_boxes[name].values())
		rows.append({
			"config": name,
			"model": model_key,
			"conf": conf,
			"imgsz": imgsz,
			"faces_detected": total_faces,
			"consensus_recall": round(matched[name] / gt_total, 3) if gt_total else None,
			"ms_per_frame": round(ms_per_frame[name], 1),
			"vs_baseline": round(ms_per_frame[name] / base_ms, 2) if base_ms else None,
		})
	return {
		"videos": [str(v) for v in videos],
		"frames_per_video": n_frames,
		"device": device,
		"consensus_gt_boxes": gt_total,
		"note": "recall은 합의 pseudo-GT 기준 상대 지표. 절대값 판단·최종 선택은 주석 프레임 육안 검증으로.",
		"results": rows,
	}


def print_table(report):
	head = f"{'config':<12} {'faces':>6} {'recall':>7} {'ms/frame':>9} {'vs_base':>8}"
	print("\n" + head + "\n" + "-" * len(head))
	for r in report["results"]:
		recall = f"{r['consensus_recall']:.3f}" if r["consensus_recall"] is not None else "-"
		vs = f"{r['vs_baseline']:.2f}x" if r["vs_baseline"] else "-"
		print(f"{r['config']:<12} {r['faces_detected']:>6} {recall:>7} {r['ms_per_frame']:>9.1f} {vs:>8}")


def smoke():
	"""합성 영상(얼굴 0개)으로 매트릭스 축소판을 완주시킨다. 지표 뼈대만 검증."""
	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "in.mp4"
		make_smoke_video(src)
		configs = [CONFIGS[0], CONFIGS[2]]  # baseline + conf-0.10
		report = run_grid([src], 4, "cpu", Path(tmp) / "vis", configs=configs)

		names = [r["config"] for r in report["results"]]
		assert names == ["baseline", "conf-0.10"], "설정 누락"
		assert all(r["ms_per_frame"] > 0 for r in report["results"]), "추론 시간 계측 실패"
		assert report["consensus_gt_boxes"] == 0, "합성 영상에 얼굴이 있을 리 없다"
		print_table(report)
		print("\nsmoke OK")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("videos", nargs="*", default=["samples/crowd.mp4", "samples/plates.mp4"],
		help="실험 대상 영상 (기본: 군중·차량 — MSG-158 완료 조건)")
	ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
	ap.add_argument("--frames", type=int, default=30, help="영상당 샘플 프레임 수")
	ap.add_argument("--out", default="results/MSG-158-grid.json")
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()

	if args.smoke:
		smoke()
		return

	report = run_grid([Path(v) for v in args.videos], args.frames, args.device,
		Path("samples/msg158"))
	Path(args.out).parent.mkdir(exist_ok=True)
	Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
	print_table(report)
	print(f"\n수치: {args.out}\n주석 프레임: samples/msg158/<config>/")


if __name__ == "__main__":
	main()
