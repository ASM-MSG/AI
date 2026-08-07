#!/usr/bin/env bash
# MSG-339 — 전용 t3.small에서 배치 크기별 처리 시간과 검출 동등성을 측정한다.
#
#   BRANCH=feature/MSG-339-ai-throughput bash ec2-bench.sh
#
# 같은 crowd 영상으로 배치 1·2·4를 각 3회 실행한다. 얼굴과 번호판은 crowd, plates,
# kr-faces, kr-plates의 동일한 30프레임을 배치 1 결과와 비교한다.

set -euo pipefail

REPO="https://github.com/ASM-MSG/AI.git"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/fillmap-bench}"
PY="$WORKDIR/.venv/bin/python"
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results}"

# MSG-207에서 기각한 실험값이 호출 셸에 남아 있어도 이번 배치 축만 움직여야 한다.
export AI_INFER_STRIDE=1
export AI_BACKEND=torch

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

log "1/5 시스템 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip ffmpeg git curl

IMDS="http://169.254.169.254/latest"
if ! IMDS_TOKEN=$(curl --noproxy '*' -sSf --max-time 2 -X PUT "$IMDS/api/token" \
	-H 'X-aws-ec2-metadata-token-ttl-seconds: 60'); then
	echo "EC2 IMDSv2를 읽지 못해 t3.small 측정인지 확인할 수 없다" >&2
	exit 1
fi
INSTANCE_TYPE=$(curl --noproxy '*' -sSf --max-time 2 \
	-H "X-aws-ec2-metadata-token: $IMDS_TOKEN" "$IMDS/meta-data/instance-type")
if [ "$INSTANCE_TYPE" != "t3.small" ]; then
	echo "t3.small 전용 측정인데 현재 인스턴스는 $INSTANCE_TYPE" >&2
	exit 1
fi

echo "arch:   $(uname -m)"
echo "cores:  $(nproc)"
echo "memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "cpu:    $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
echo "instance: $INSTANCE_TYPE"

log "2/5 레포 준비 (branch: $BRANCH)"
if [ -d "$WORKDIR/.git" ]; then
	git -C "$WORKDIR" fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
	git -C "$WORKDIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
	git clone --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

log "3/5 파이썬 의존성"
python3 -m venv --clear .venv
"$WORKDIR/.venv/bin/pip" install -q --upgrade pip
if [ "$(uname -m)" = "x86_64" ]; then
	"$WORKDIR/.venv/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
"$WORKDIR/.venv/bin/pip" install -q -r requirements.txt

log "4/5 검증 영상"
mkdir -p samples "$RESULTS_DIR"
fetch() {
	local pexels_id=$1 name=$2
	[ -s "samples/$name.mp4" ] && { echo "  $name.mp4 (캐시됨)"; return; }
	curl -sSfL -o "/tmp/msg339-$name-raw.mp4" "https://www.pexels.com/download/video/$pexels_id/"
	ffmpeg -y -v error -stream_loop -1 -i "/tmp/msg339-$name-raw.mp4" -t 29 -an \
		-c:v libx264 -preset veryfast -pix_fmt yuv420p "samples/$name.mp4"
	rm -f "/tmp/msg339-$name-raw.mp4"
	echo "  $name.mp4"
}
fetch 855564 crowd
fetch 854671 plates

for sample in kr-faces kr-plates; do
	if [ ! -s "samples/$sample.mp4" ]; then
		cat >&2 <<EOF
한국 검증 영상 samples/$sample.mp4가 없어 중단한다.
로컬의 비식별 테스트 클립을 아래처럼 전용 EC2에 복사한 뒤 다시 실행한다.
  scp samples/$sample.mp4 ubuntu@<AI_EIP>:$WORKDIR/samples/$sample.mp4
실제 얼굴·번호판 영상은 samples/가 gitignore 대상이므로 커밋하지 않는다.
EOF
		exit 1
	fi
done

log "5/5 배치 1·2·4 단일 작업 각 3회"
for batch in 1 2 4; do
	for run in 1 2 3; do
		result="$RESULTS_DIR/MSG-339-bench-b${batch}-r${run}.json"
		echo "  batch=$batch run=$run"
		AI_BATCH_SIZE="$batch" "$PY" bench.py samples/crowd.mp4 --device cpu \
			--out "/tmp/msg339-crowd-b${batch}-r${run}.mp4" > "$result"
	done
done

"$PY" - "$RESULTS_DIR" "$INSTANCE_TYPE" <<'PY'
import glob
import json
import statistics
import sys
from pathlib import Path

results_dir, instance_type = Path(sys.argv[1]), sys.argv[2]
rows = []
for batch in (1, 2, 4):
	runs = [json.loads(Path(path).read_text())
		for path in sorted(glob.glob(str(results_dir / f"MSG-339-bench-b{batch}-r*.json")))]
	if len(runs) != 3:
		raise SystemExit(f"batch {batch}: 3회 결과가 필요하지만 {len(runs)}개")
	rows.append({
		"batch_size": batch,
		"runs": [{"wall_sec": run["wall_sec"], "peak_memory_mb": run["peak_memory_mb"]} for run in runs],
		"median_wall_sec": round(statistics.median(run["wall_sec"] for run in runs), 3),
		"peak_memory_mb": max(run["peak_memory_mb"] for run in runs),
	})

out = results_dir / "MSG-339-bench-summary.json"
out.write_text(json.dumps({"instance_type": instance_type, "backend": "torch", "infer_stride": 1,
	"video": "samples/crowd.mp4", "runs_per_batch": 3, "results": rows},
	indent=2, ensure_ascii=False) + "\n")
print(out.read_text())
PY

log "검출 동등성: 배치 1 기준 IoU 0.5 상대 recall"
"$PY" - "$RESULTS_DIR/MSG-339-detections.json" <<'PY'
import json
import sys
from pathlib import Path

from face_experiment import iou
from bench import FACE_CONF, PLATE_CONF, load_models, sample_frames, to_boxes

videos = [Path("samples/crowd.mp4"), Path("samples/plates.mp4"),
	Path("samples/kr-faces.mp4"), Path("samples/kr-plates.mp4")]
frames_by_video = {str(video): sample_frames(video, 30) for video in videos}
for video, frames in frames_by_video.items():
	if len(frames) != 30:
		raise SystemExit(f"{video}: 30프레임이 필요하지만 {len(frames)}장만 읽음")


def detect(batch_size):
	face, plate = load_models("cpu")
	out = {"face": {}, "plate": {}}
	for video, indexed_frames in frames_by_video.items():
		for start in range(0, len(indexed_frames), batch_size):
			chunk = indexed_frames[start:start + batch_size]
			images = [frame for _, frame in chunk]
			face_results = face.predict(images, conf=FACE_CONF, verbose=False)
			plate_results = plate.predict(images, conf=PLATE_CONF, verbose=False)
			for (frame_index, frame), face_result, plate_result in zip(chunk, face_results, plate_results):
				height, width = frame.shape[:2]
				key = f"{video}:{frame_index}"
				out["face"][key] = to_boxes(face_result, width, height)
				out["plate"][key] = to_boxes(plate_result, width, height)
	del face, plate
	return out


def relative_recall(baseline, candidate):
	total = matched = 0
	for key, expected_boxes in baseline.items():
		actual_boxes = candidate.get(key, [])
		used = set()
		for expected in expected_boxes:
			total += 1
			matches = [(iou(expected, actual), index) for index, actual in enumerate(actual_boxes)
				if index not in used and iou(expected, actual) >= 0.5]
			if matches:
				_, index = max(matches)
				used.add(index)
				matched += 1
	return {"baseline_boxes": total, "matched_boxes": matched,
		"relative_recall": round(matched / total, 3) if total else None}


boxes = {str(batch): detect(batch) for batch in (1, 2, 4)}
comparisons = []
passed = True
for batch in (2, 4):
	row = {"batch_size": batch}
	for kind in ("face", "plate"):
		metric = relative_recall(boxes["1"][kind], boxes[str(batch)][kind])
		row[kind] = metric
		passed = passed and metric["baseline_boxes"] > 0 and metric["matched_boxes"] == metric["baseline_boxes"]
	comparisons.append(row)

report = {"backend": "torch", "infer_stride": 1, "videos": [str(video) for video in videos],
	"frames_per_video": 30, "iou_threshold": 0.5, "boxes": boxes,
	"comparisons": comparisons, "passed": passed}
out = Path(sys.argv[1])
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"comparisons": comparisons, "passed": passed}, indent=2, ensure_ascii=False))
if not passed:
	raise SystemExit("얼굴 또는 번호판 상대 recall 1.000 기준 미달")
PY

log "완료"
echo "단일 작업 요약: $RESULTS_DIR/MSG-339-bench-summary.json"
echo "검출 원자료:    $RESULTS_DIR/MSG-339-detections.json"
