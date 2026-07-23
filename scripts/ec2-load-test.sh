#!/usr/bin/env bash
# MSG-151 — dev 파이프라인(BE→AI) 동시 업로드 부하 측정. EC2에서 실행.
#
#   N=3 bash ec2-load-test.sh        # 동시 3건 (기본값 3)
#
# 산출물: /tmp/msg151/N{N}-{시각}/
#   status.csv     15초 폴링 상태 스냅샷 (epoch,videoId,processing_status)
#   resources.csv  10초 리소스 샘플 (epoch,used_MB,avail_MB,swap_used_MB,load1)
#   final.csv      DB 최종 상태 (blurring_started_at 포함 — 정확한 전이 시각)
#   summary.txt    영상별 구간 소요·최종 상태·리소스 피크
#
# 유의: dev BE(localhost:8080)에 실제 부하를 건다 — 팀이 dev를 쓰는 시간대는 피할 것.
set -euo pipefail
N="${N:-3}"
BASE=localhost:8080
SAMPLE=/tmp/msg151-crowd.mp4
OUT="/tmp/msg151/N$N-$(date +%m%d-%H%M%S)"
mkdir -p "$OUT"
PSQL="sudo docker exec fillmap-postgres-dev psql -U dev -d fillmap -t -A -F,"

log() { echo -e "\033[1;36m=== $*\033[0m"; }
pyget() { python3 -c "import json,sys; d=json.load(sys.stdin); d=d.get('body', d); print(d['$1'])"; }

# ---------------------------------------------------------------- 0. 샘플
# MSG-142 crowd.mp4 (pexels 855564, 회랑에 사람 빽빽) — '사람 많은 30초 영상' 조건 그대로.
log "0/4 샘플 준비"
if [ ! -s "$SAMPLE" ]; then
	curl -sL -o /tmp/raw-crowd.mp4 "https://www.pexels.com/download/video/855564/"
	# -t 30은 프레임 경계 때문에 30.03s가 나와 BE 30초 상한 검증에 걸린다 → 29초로 자른다
	ffmpeg -y -v error -stream_loop -1 -i /tmp/raw-crowd.mp4 -t 29 \
		-vf "scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2" \
		-c:v libx264 -preset veryfast -pix_fmt yuv420p "$SAMPLE"
	rm -f /tmp/raw-crowd.mp4
fi
ls -lh "$SAMPLE"

TOKEN=$(curl -sf -X POST $BASE/api/auth/dev/social-login -H 'Content-Type: application/json' \
	-d '{"provider":"KAKAO","oid":"msg151-load"}' | pyget accessToken)
SIZE=$(stat -c%s "$SAMPLE")

# ---------------------------------------------------------------- 1. 리소스 샘플러 (10초)
( while :; do
	echo "$(date +%s),$(free -m | awk '/Mem:/{printf "%s,%s", $3, $7}'),$(awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%.0f", (t-f)/1024}' /proc/meminfo),$(cut -d' ' -f1 /proc/loadavg)"
	sleep 10
done >> "$OUT/resources.csv" ) &
RES_PID=$!
trap 'kill $RES_PID 2>/dev/null || true' EXIT

# ---------------------------------------------------------------- 2. N건 동시 업로드
log "1/4 동시 $N건 업로드"
upload_one() {
	local i=$1 t0 t1 pres url key lat vid
	t0=$(date +%s.%N)
	pres=$(curl -sf -X POST $BASE/api/videos/presigned-url -H "Authorization: Bearer $TOKEN" \
		-H 'Content-Type: application/json' \
		-d "{\"extension\":\"mp4\",\"contentType\":\"video/mp4\",\"contentLength\":$SIZE}")
	url=$(echo "$pres" | pyget uploadUrl); key=$(echo "$pres" | pyget s3Key)
	curl -sf -X PUT --upload-file "$SAMPLE" -H 'Content-Type: video/mp4' "$url" >/dev/null
	# 격자 점령 경합이 측정 노이즈가 되지 않게 건마다 다른 격자 좌표
	lat=$(awk "BEGIN{printf \"%.2f\", 37.50 + $i * 0.01}")
	vid=$(curl -sf -X POST $BASE/api/videos -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
		-d "{\"s3Key\":\"$key\",\"lat\":$lat,\"lon\":126.9780,\"durationSec\":29,\"recordedAt\":\"2026-07-22T16:00:00\"}" | pyget videoId)
	t1=$(date +%s.%N)
	echo "$vid,$t0,$t1" > "$OUT/upload-$i.csv"
	echo "  #$i → videoId=$vid ($(awk "BEGIN{printf \"%.1f\", $t1-$t0}")s)"
}
# 인자 없는 wait는 리소스 샘플러(무한 루프)까지 기다린다 — 업로드 PID만 명시
PIDS=()
for i in $(seq 1 "$N"); do upload_one "$i" & PIDS+=($!); done
wait "${PIDS[@]}"
IDS=$(cut -d, -f1 "$OUT"/upload-*.csv | paste -sd, -)
echo "videoIds: $IDS"

# ---------------------------------------------------------------- 3. 폴링 (15초)
log "2/4 폴링 — 전부 READY/FAILED까지"
START=$(date +%s)
while :; do
	NOW=$(date +%s)
	ROWS=$($PSQL -c "SELECT id, processing_status FROM videos WHERE id IN ($IDS)")
	echo "$ROWS" | while IFS=, read -r id st; do echo "$NOW,$id,$st"; done >> "$OUT/status.csv"
	LEFT=$(echo "$ROWS" | grep -cvE "READY|FAILED" || true)
	echo "  [$((NOW-START))s] 남은 $LEFT건 :: $(echo "$ROWS" | tr '\n' ' ')"
	[ "$LEFT" -eq 0 ] && break
	[ $((NOW-START)) -gt 5400 ] && { echo "90분 초과 — 중단"; break; }
	sleep 15
done
kill $RES_PID 2>/dev/null || true

# ---------------------------------------------------------------- 4. 요약
log "3/4 DB 최종 상태"
$PSQL -c "SELECT id, processing_status, blurring_started_at, created_at FROM videos WHERE id IN ($IDS) ORDER BY id" > "$OUT/final.csv"
cat "$OUT/final.csv"

log "4/4 요약"
python3 - "$OUT" <<'PY'
import csv, sys, collections, os
out = sys.argv[1]
first = collections.defaultdict(dict)          # 상태별 최초 관측 시각 (15초 해상도)
for ts, vid, st in csv.reader(open(f"{out}/status.csv")):
    first[vid].setdefault(st, int(ts))
t0map = {}
for f in os.listdir(out):
    if f.startswith("upload-"):
        vid, t0, t1 = open(f"{out}/{f}").read().strip().split(",")
        t0map[vid] = (float(t0), float(t1))
lines = ["video,confirm_s,encode_wait+run_s,blur_queue+run_s,total_s,final"]
for vid, states in sorted(first.items(), key=lambda kv: int(kv[0])):
    t0, t1 = t0map.get(vid, (None, None))
    fin = "READY" if "READY" in states else ("FAILED" if "FAILED" in states else "STUCK")
    end = states.get("READY") or states.get("FAILED")
    blur = states.get("BLURRING")
    fmt = lambda v: round(v, 1) if v is not None else ""
    lines.append(f"{vid},{fmt(t1 - t0 if t0 else None)},"
                 f"{fmt(blur - t1 if blur and t1 else None)},"
                 f"{fmt(end - blur if end and blur else None)},"
                 f"{fmt(end - t1 if end and t1 else None)},{fin}")
res = list(csv.reader(open(f"{out}/resources.csv")))
if res:
    lines.append(f"# mem used peak {max(int(r[1]) for r in res)}MB / avail min {min(int(r[2]) for r in res)}MB"
                 f" / swap peak {max(int(r[3]) for r in res)}MB / load1 peak {max(float(r[4]) for r in res)}")
open(f"{out}/summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

log "완료 — 결과: $OUT"
