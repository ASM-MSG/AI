#!/usr/bin/env bash
# MSG-161 — dev EC2에 AI 서버를 Docker로 배포하고 실영상 E2E로 검증한다.
#
#   scp scripts/ec2-deploy.sh ubuntu@<IP>:~/
#   ssh ubuntu@<IP> 'bash ec2-deploy.sh'
#
# 재실행하면 기존 컨테이너를 교체한다. 모델 가중치는 hf-cache 볼륨에 남아
# 재배포 때 다시 받지 않는다.

set -euo pipefail

REPO="https://github.com/ASM-MSG/AI.git"
BRANCH="${BRANCH:-main}"
WORKDIR="$HOME/fillmap-ai"
PORT="${PORT:-8000}"
ENV_FILE="$HOME/fillmap-ai.env"

if [ -f "$ENV_FILE" ]; then
	set -a
	# shellcheck source=/dev/null
	. "$ENV_FILE"
	set +a
fi
AI_WORKERS="${AI_WORKERS:-1}"
AI_BATCH_SIZE="${AI_BATCH_SIZE:-1}"
case "$AI_WORKERS" in 1|2) ;; *) echo "AI_WORKERS는 1 또는 2만 허용" >&2; exit 1 ;; esac
case "$AI_BATCH_SIZE" in 1|2|4) ;; *) echo "AI_BATCH_SIZE는 1, 2, 4만 허용" >&2; exit 1 ;; esac

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

# ---------------------------------------------------------------- 1. Docker
log "1/5 Docker"
if ! command -v docker >/dev/null; then
	sudo apt-get update -qq
	sudo apt-get install -y -qq docker.io
fi
sudo docker version --format '  {{.Server.Os}}/{{.Server.Arch}}'

# ---------------------------------------------------------------- 2. 레포
log "2/5 레포 준비 (branch: $BRANCH)"
if [ -d "$WORKDIR/.git" ]; then
	git -C "$WORKDIR" fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
	git -C "$WORKDIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
	git clone --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

# ---------------------------------------------------------------- 3. 빌드·기동
log "3/5 이미지 빌드"
sudo docker build -t fillmap-ai .

log "4/5 컨테이너 교체"
sudo docker rm -f fillmap-ai 2>/dev/null || true
sudo docker run -d --name fillmap-ai --restart unless-stopped \
	-p "$PORT:8000" -v hf-cache:/root/.cache/huggingface \
	-e AI_WORKERS="$AI_WORKERS" -e AI_BATCH_SIZE="$AI_BATCH_SIZE" fillmap-ai
echo "  AI_WORKERS=$AI_WORKERS / AI_BATCH_SIZE=$AI_BATCH_SIZE"

echo -n "  /health 대기"
for _ in $(seq 1 30); do
	curl -sf "localhost:$PORT/health" >/dev/null && break
	echo -n "."; sleep 2
done
curl -sf "localhost:$PORT/health" || { echo "서버가 뜨지 않음"; sudo docker logs fillmap-ai | tail -20; exit 1; }
echo

# ---------------------------------------------------------------- 4. E2E
# ec2-bench.sh와 같은 실영상(plates, 1080p 30초) 2건으로 업로드→폴링→다운로드 왕복.
log "5/5 실영상 2건 동시 E2E (1080p 30초)"
SAMPLE=/tmp/e2e-plates.mp4
if [ ! -s "$SAMPLE" ]; then
	curl -sL -o /tmp/raw-e2e.mp4 "https://www.pexels.com/download/video/854671/"
	# ffmpeg은 방금 빌드한 이미지 안의 것을 쓴다 — 호스트에 따로 깔지 않는다.
	# 호스트 ffmpeg에 의존했다가 새로 만든 EC2에 없어서 여기서 죽은 적이 있다 (MSG-282).
	sudo docker run --rm -v /tmp:/tmp fillmap-ai \
		ffmpeg -y -v error -stream_loop -1 -i /tmp/raw-e2e.mp4 -t 30 \
		-c:v libx264 -preset veryfast -pix_fmt yuv420p "$SAMPLE"
	rm -f /tmp/raw-e2e.mp4
fi

json_field() { python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
PIDS=()
cleanup() {
	local pid
	for pid in "${PIDS[@]}"; do
		kill "$pid" 2>/dev/null || true
	done
	for pid in "${PIDS[@]}"; do
		wait "$pid" 2>/dev/null || true
	done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

START=$(date +%s)
submit_job() {
	local index=$1 response job_id
	response=$(curl -sSf -F "file=@$SAMPLE;type=video/mp4" "localhost:$PORT/jobs")
	job_id=$(printf '%s' "$response" | json_field job_id)
	printf '%s\n' "$job_id" > "/tmp/e2e-job-$index.id"
}

for index in 1 2; do
	submit_job "$index" &
	PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
	wait "$pid"
done
PIDS=()

JOBS=()
for index in 1 2; do
	read -r job_id < "/tmp/e2e-job-$index.id"
	JOBS+=("$job_id")
	echo "  #$index job_id: $job_id"
done

declare -A FINAL_STATUS
while :; do
	ELAPSED=$(( $(date +%s) - START ))
	remaining=0
	for index in 0 1; do
		job_id="${JOBS[$index]}"
		if [ -n "${FINAL_STATUS[$job_id]:-}" ]; then
			continue
		fi
		BODY=$(curl -sSf "localhost:$PORT/jobs/$job_id")
		STATUS=$(printf '%s' "$BODY" | json_field status)
		echo "  [${ELAPSED}s] #$((index + 1)) $STATUS"
		case "$STATUS" in
			DONE) FINAL_STATUS[$job_id]="DONE" ;;
			FAILED)
				echo "$BODY"
				sudo docker logs fillmap-ai | tail -20
				exit 1
				;;
			*) remaining=$((remaining + 1)) ;;
		esac
	done
	[ "$remaining" -eq 0 ] && break
	[ "$ELAPSED" -gt 900 ] && { echo "15분 초과 — 실패로 간주"; exit 1; }
	sleep 10
done

for index in 0 1; do
	job_id="${JOBS[$index]}"
	out="/tmp/e2e-out-$((index + 1)).mp4"
	curl -sSf -o "$out" "localhost:$PORT/jobs/$job_id/video"
	sudo docker run --rm -v /tmp:/tmp:ro fillmap-ai ffprobe -v error \
		-select_streams v:0 -show_entries stream=codec_name:format=duration,size \
		-of json "$out" > "/tmp/e2e-ffprobe-$((index + 1)).json"
	codec=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["streams"][0]["codec_name"])' \
		< "/tmp/e2e-ffprobe-$((index + 1)).json")
	[ "$codec" = "h264" ] || { echo "결과 코덱이 h264가 아니다: $codec" >&2; exit 1; }
	ls -lh "$out"
done

log "완료 — 처리 $(( $(date +%s) - START ))초. 결과: /tmp/e2e-out-{1,2}.mp4"
