#!/usr/bin/env bash
# MSG-339 — 분리된 AI API에 직접 N건을 제출해 처리량과 자원을 측정한다.
#
#   N=6 BASE=http://localhost:8000 bash ec2-ai-load-test.sh
#
# 산출물은 /tmp/msg339-ai/N{N}-{시각}/에 남는다. 상태는 2초, 메모리·스왑·si·so·load1은
# 10초 간격으로 기록한다. 실행 중인 fillmap-ai 컨테이너와 이미지가 필요하다.

set -euo pipefail

N="${N:-6}"
BASE="${BASE:-http://localhost:8000}"
SAMPLE="${SAMPLE:-/tmp/msg339-crowd.mp4}"
OUT="${OUT:-/tmp/msg339-ai/N${N}-$(date +%m%d-%H%M%S)}"
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
CONTAINER="${CONTAINER:-fillmap-ai}"
IMAGE="${IMAGE:-fillmap-ai}"

case "$N" in ''|*[!0-9]*) echo "N은 양의 정수여야 한다" >&2; exit 1 ;; esac
[ "$N" -gt 0 ] || { echo "N은 1 이상이어야 한다" >&2; exit 1; }
FAILED=0

mkdir -p "$OUT"

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }
json_field() { python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

BG_PIDS=()
cleanup() {
	local pid
	for pid in "${BG_PIDS[@]}"; do
		kill "$pid" 2>/dev/null || true
	done
	for pid in "${BG_PIDS[@]}"; do
		wait "$pid" 2>/dev/null || true
	done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "0/4 사전 확인"
curl -sSf "$BASE/health" >/dev/null
if [ ! -s "$SAMPLE" ]; then
	curl -sSfL -o /tmp/msg339-crowd-raw.mp4 "https://www.pexels.com/download/video/855564/"
	sudo docker run --rm -v /tmp:/tmp "$IMAGE" ffmpeg -y -v error -stream_loop -1 \
		-i /tmp/msg339-crowd-raw.mp4 -t 29 -an -c:v libx264 -preset veryfast \
		-pix_fmt yuv420p "$SAMPLE"
	rm -f /tmp/msg339-crowd-raw.mp4
fi
ls -lh "$SAMPLE"

read -r RESTART_BEFORE OOM_BEFORE < <(
	sudo docker inspect -f '{{.RestartCount}} {{.State.OOMKilled}}' "$CONTAINER"
)

echo "epoch,mem_used_mb,mem_available_mb,swap_used_mb,si,so,load1" > "$OUT/resources.csv"
resource_sampler() {
	local memory swap_used si so load1
	while :; do
		read -r si so < <(vmstat 10 2 | awk 'END{print $7, $8}')
		memory=$(free -m | awk '/^Mem:/{print $3","$7}')
		swap_used=$(awk '/SwapTotal/{total=$2} /SwapFree/{free=$2} END{printf "%.0f", (total-free)/1024}' \
			/proc/meminfo)
		load1=$(cut -d' ' -f1 /proc/loadavg)
		echo "$(date +%s),$memory,$swap_used,$si,$so,$load1" >> "$OUT/resources.csv"
	done
}
resource_sampler &
RES_PID=$!
BG_PIDS+=("$RES_PID")

log "1/4 동시 $N건 제출"
submit_one() {
	local index=$1 response job_id queued_at
	response=$(curl -sSf -F "file=@$SAMPLE;type=video/mp4" "$BASE/jobs")
	job_id=$(printf '%s' "$response" | json_field job_id)
	queued_at=$(date +%s)
	printf '%s,%s,%s\n' "$index" "$job_id" "$queued_at" > "$OUT/upload-$index.csv"
}

UPLOAD_PIDS=()
for ((index = 1; index <= N; index++)); do
	submit_one "$index" &
	UPLOAD_PIDS+=("$!")
	BG_PIDS+=("$!")
done
for pid in "${UPLOAD_PIDS[@]}"; do
	wait "$pid"
done
BG_PIDS=("$RES_PID")

echo "index,job_id,queued_at" > "$OUT/jobs.csv"
echo "epoch,job_id,status" > "$OUT/status.csv"
for ((index = 1; index <= N; index++)); do
	IFS=, read -r _ job_id queued_at < "$OUT/upload-$index.csv"
	echo "$index,$job_id,$queued_at" >> "$OUT/jobs.csv"
	echo "$queued_at,$job_id,QUEUED" >> "$OUT/status.csv"
	echo "  #$index job_id=$job_id"
done

log "2/4 상태 폴링"
declare -A FINAL_STATUS
START=$(date +%s)
while :; do
	remaining=0
	now=$(date +%s)
	for ((index = 1; index <= N; index++)); do
		IFS=, read -r _ job_id _ < "$OUT/upload-$index.csv"
		if [ -n "${FINAL_STATUS[$job_id]:-}" ]; then
			continue
		fi
		body=$(curl -sSf "$BASE/jobs/$job_id")
		status=$(printf '%s' "$body" | json_field status)
		echo "$now,$job_id,$status" >> "$OUT/status.csv"
		case "$status" in
			DONE|FAILED) FINAL_STATUS[$job_id]="$status" ;;
			*) remaining=$((remaining + 1)) ;;
		esac
	done
	echo "  [$((now - START))s] 남은 $remaining건"
	[ "$remaining" -eq 0 ] && break
	if [ $((now - START)) -gt "$TIMEOUT_SEC" ]; then
		echo "${TIMEOUT_SEC}초 초과 — 남은 작업을 STUCK으로 판정" >&2
		break
	fi
	sleep 2
done

if ! kill -0 "$RES_PID" 2>/dev/null; then
	echo "자원 샘플러가 측정 중 종료됐다" >&2
	FAILED=1
fi
kill "$RES_PID" 2>/dev/null || true
wait "$RES_PID" 2>/dev/null || true
BG_PIDS=()

log "3/4 결과 영상과 컨테이너 상태"
for ((index = 1; index <= N; index++)); do
	IFS=, read -r _ job_id _ < "$OUT/upload-$index.csv"
	status="${FINAL_STATUS[$job_id]:-STUCK}"
	if [ "$status" != "DONE" ]; then
		echo "  #$index $job_id: $status" >&2
		FAILED=1
		continue
	fi
	video="$OUT/result-$index.mp4"
	if ! curl -sSf -o "$video" "$BASE/jobs/$job_id/video"; then
		echo "  #$index 결과 다운로드 실패" >&2
		FAILED=1
		continue
	fi
	if command -v ffprobe >/dev/null; then
		if ! ffprobe -v error -select_streams v:0 -show_entries stream=codec_name:format=duration,size \
			-of json "$video" > "$OUT/ffprobe-$index.json"; then
			FAILED=1
		fi
	elif ! sudo docker run --rm -v "$OUT:/results:ro" "$IMAGE" ffprobe -v error \
		-select_streams v:0 -show_entries stream=codec_name:format=duration,size -of json \
		"/results/result-$index.mp4" > "$OUT/ffprobe-$index.json"; then
		FAILED=1
	fi
	if ! codec=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["streams"][0]["codec_name"])' \
		< "$OUT/ffprobe-$index.json"); then
		FAILED=1
	elif [ "$codec" != "h264" ]; then
		echo "  #$index 결과 코덱이 h264가 아니다: $codec" >&2
		FAILED=1
	fi
done

read -r RESTART_AFTER OOM_AFTER < <(
	sudo docker inspect -f '{{.RestartCount}} {{.State.OOMKilled}}' "$CONTAINER"
)
if [ "$RESTART_AFTER" -ne "$RESTART_BEFORE" ] || [ "$OOM_AFTER" = "true" ]; then
	FAILED=1
fi

log "4/4 요약"
if ! python3 - "$OUT" "$RESTART_BEFORE" "$RESTART_AFTER" "$OOM_BEFORE" "$OOM_AFTER" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
restart_before, restart_after, oom_before, oom_after = sys.argv[2:]
first = defaultdict(dict)
last = {}
with (out / "status.csv").open() as file:
	for row in csv.DictReader(file):
		first[row["job_id"]].setdefault(row["status"], int(row["epoch"]))
		last[row["job_id"]] = row["status"]

lines = ["job_id,queued_at,processing_at,done_at,queue_s,processing_s,total_s,result_size_bytes,final"]
queued_times = []
done_times = []
with (out / "jobs.csv").open() as file:
	for row in csv.DictReader(file):
		index = row["index"]
		job_id = row["job_id"]
		queued = int(row["queued_at"])
		processing = first[job_id].get("PROCESSING")
		done = first[job_id].get("DONE")
		queued_times.append(queued)
		if done is not None:
			done_times.append(done)
		result = out / f"result-{index}.mp4"
		value = lambda item: "" if item is None else item
		elapsed = lambda end, start: "" if end is None else end - start
		final = last.get(job_id, "STUCK")
		final = final if final in ("DONE", "FAILED") else "STUCK"
		lines.append(f"{job_id},{queued},{value(processing)},{value(done)},"
			f"{elapsed(processing, queued)},{elapsed(done, processing) if processing else ''},"
			f"{elapsed(done, queued)},{result.stat().st_size if result.exists() else ''},{final}")

resources = list(csv.DictReader((out / "resources.csv").open()))
peak_consecutive = 0
if resources:
	current = 0
	for row in resources:
		current = current + 1 if int(row["si"]) > 0 or int(row["so"]) > 0 else 0
		peak_consecutive = max(peak_consecutive, current)
	lines += [
		f"# mem_used_peak_mb={max(int(row['mem_used_mb']) for row in resources)}",
		f"# mem_available_min_mb={min(int(row['mem_available_mb']) for row in resources)}",
		f"# swap_used_peak_mb={max(int(row['swap_used_mb']) for row in resources)}",
		f"# load1_peak={max(float(row['load1']) for row in resources):.2f}",
		f"# swap_io_consecutive_peak={peak_consecutive}",
	]
makespan = max(done_times) - min(queued_times) if done_times and queued_times else None
throughput = len(done_times) * 3600 / makespan if makespan else None
lines += [f"# completed_jobs={len(done_times)}", f"# makespan_s={makespan if makespan is not None else ''}",
	f"# throughput_per_hour={throughput:.2f}" if throughput is not None else "# throughput_per_hour="]
lines += [f"# docker_restart_count={restart_before}->{restart_after}",
	f"# docker_oom_killed={oom_before}->{oom_after}"]
(out / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if not resources:
	print("자원 표본이 1개도 없다", file=sys.stderr)
	raise SystemExit(1)
if peak_consecutive >= 3:
	print(f"swap-in 또는 swap-out 연속 발생 {peak_consecutive}회", file=sys.stderr)
	raise SystemExit(1)
PY
	then
	FAILED=1
fi

echo "결과 디렉터리: $OUT"
echo "요약: $OUT/summary.txt"
if [ "$FAILED" -ne 0 ]; then
	echo "DONE·다운로드·ffprobe·컨테이너 안정성 중 실패가 있다" >&2
	exit 1
fi
