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
	git -C "$WORKDIR" fetch origin "$BRANCH" && git -C "$WORKDIR" checkout "$BRANCH" && git -C "$WORKDIR" pull --ff-only
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
	-p "$PORT:8000" -v hf-cache:/root/.cache/huggingface fillmap-ai

echo -n "  /health 대기"
for _ in $(seq 1 30); do
	curl -sf "localhost:$PORT/health" >/dev/null && break
	echo -n "."; sleep 2
done
curl -sf "localhost:$PORT/health" || { echo "서버가 뜨지 않음"; sudo docker logs fillmap-ai | tail -20; exit 1; }
echo

# ---------------------------------------------------------------- 4. E2E
# ec2-bench.sh와 같은 실영상(plates, 1080p 30초)으로 업로드→폴링→다운로드 왕복.
log "5/5 실영상 E2E (1080p 30초 — 3~4분 예상)"
SAMPLE=/tmp/e2e-plates.mp4
if [ ! -s "$SAMPLE" ]; then
	curl -sL -o /tmp/raw-e2e.mp4 "https://www.pexels.com/download/video/854671/"
	ffmpeg -y -v error -stream_loop -1 -i /tmp/raw-e2e.mp4 -t 30 \
		-c:v libx264 -preset veryfast -pix_fmt yuv420p "$SAMPLE"
	rm -f /tmp/raw-e2e.mp4
fi

START=$(date +%s)
JOB=$(curl -sf -F "file=@$SAMPLE;type=video/mp4" "localhost:$PORT/jobs" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
echo "  job_id: $JOB"

while :; do
	BODY=$(curl -sf "localhost:$PORT/jobs/$JOB")
	STATUS=$(echo "$BODY" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
	ELAPSED=$(( $(date +%s) - START ))
	echo "  [${ELAPSED}s] $STATUS"
	case "$STATUS" in
		DONE) break ;;
		FAILED) echo "$BODY"; sudo docker logs fillmap-ai | tail -20; exit 1 ;;
	esac
	[ "$ELAPSED" -gt 900 ] && { echo "15분 초과 — 실패로 간주"; exit 1; }
	sleep 15
done

echo "  응답: $BODY"
curl -sf -o /tmp/e2e-out.mp4 "localhost:$PORT/jobs/$JOB/video"
ls -lh /tmp/e2e-out.mp4

log "완료 — 처리 $(( $(date +%s) - START ))초. 블러 확인: scp ubuntu@<IP>:/tmp/e2e-out.mp4 ."
