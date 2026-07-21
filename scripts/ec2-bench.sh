#!/usr/bin/env bash
# MSG-142 — EC2에서 블러 파이프라인 처리 시간을 측정한다.
#
# 로컬(Apple M5) 측정치는 인스턴스보다 훨씬 빨라 인프라 판단에 쓸 수 없다.
# 이 스크립트는 실제 리눅스 인스턴스에서 같은 파이프라인을 돌려 MSG-143의 근거를 만든다.
#
# 권장 인스턴스: c7g.large (2 vCPU / 4GB, arm64)
#   - 1 vCPU 측정은 taskset으로 코어를 묶어서 얻는다 (Lambda 1769MB = 1 vCPU 상당)
#   - c7g.medium(1 vCPU / 2GB)은 4K 처리 시 메모리가 빠듯하다 (로컬 피크 897MB + torch 오버헤드)
# AMI: Ubuntu 24.04 LTS
#
#   scp scripts/ec2-bench.sh ubuntu@<IP>:~/
#   ssh ubuntu@<IP> 'bash ec2-bench.sh'

set -euo pipefail

REPO="https://github.com/ASM-MSG/AI.git"
BRANCH="${BRANCH:-main}"   # bench.py 가 아직 머지 전이면 작업 브랜치를 넘긴다
WORKDIR="$HOME/fillmap-bench"
PY="$WORKDIR/.venv/bin/python"

log() { echo -e "\n\033[1;36m=== $* ===\033[0m"; }

# ---------------------------------------------------------------- 1. 시스템
log "1/6 시스템 패키지"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip ffmpeg git curl

log "환경"
echo "arch:   $(uname -m)"
echo "cores:  $(nproc)"
echo "memory: $(free -h | awk '/^Mem:/{print $2}')"
echo "cpu:    $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"

# ---------------------------------------------------------------- 2. 레포
log "2/6 레포 준비"
if [ -d "$WORKDIR/.git" ]; then
	git -C "$WORKDIR" pull --ff-only
else
	git clone --depth 1 -b "$BRANCH" "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"

# ---------------------------------------------------------------- 3. 파이썬
log "3/6 파이썬 의존성 (수 분 소요)"
python3 -m venv .venv
"$WORKDIR/.venv/bin/pip" install -q --upgrade pip
# x86_64는 기본 wheel이 CUDA를 끌고 와 2GB+ 받는다. CPU 전용 인덱스를 쓴다.
# arm64(Graviton)는 기본 wheel이 이미 CPU 전용이다.
if [ "$(uname -m)" = "x86_64" ]; then
	"$WORKDIR/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
"$WORKDIR/.venv/bin/pip" install -q -r requirements.txt

# ---------------------------------------------------------------- 4. 샘플
# 로컬 측정과 같은 영상을 써야 비교가 성립한다. 원본이 12~27초라 30초로 늘린다.
log "4/6 샘플 영상"
mkdir -p samples
fetch() { # fetch <pexels_id> <이름>
	[ -s "samples/$2.mp4" ] && { echo "  $2.mp4 (캐시됨)"; return; }
	curl -sL -o "/tmp/raw-$1.mp4" "https://www.pexels.com/download/video/$1/"
	ffmpeg -y -v error -stream_loop -1 -i "/tmp/raw-$1.mp4" -t 30 -an \
		-c:v libx264 -preset veryfast -pix_fmt yuv420p "samples/$2.mp4"
	rm -f "/tmp/raw-$1.mp4"
	echo "  $2.mp4"
}
fetch 5021553  normal   # 사람 몇 명이 지나가는 거리 (1080p)
fetch 855564   crowd    # 회랑 군중 (1080p)
fetch 854671   plates   # 도로 정면, 번호판 선명 (1080p)
fetch 12699538 hires    # 군중, 4K 세로 60fps

# ---------------------------------------------------------------- 5. 검증
# macOS에서는 torch 스레드 제한이 Accelerate/AMX 백엔드에 무시돼 측정이 무효였다.
# 리눅스에서 taskset이 실제로 먹는지 먼저 확인하고, 아니면 1 vCPU 측정을 건너뛴다.
log "5/6 코어 제한 유효성 검증"
matmul() { "$@" "$PY" -c "
import torch, time
a = torch.randn(2000, 2000); b = torch.randn(2000, 2000)
torch.mm(a, b)
s = time.perf_counter()
for _ in range(20): torch.mm(a, b)
print(f'{time.perf_counter() - s:.3f}')
"; }

FULL=$(matmul)
ONE=$(taskset -c 0 "$PY" -c "
import torch, time
torch.set_num_threads(1)
a = torch.randn(2000, 2000); b = torch.randn(2000, 2000)
torch.mm(a, b)
s = time.perf_counter()
for _ in range(20): torch.mm(a, b)
print(f'{time.perf_counter() - s:.3f}')
")
echo "  matmul 20회 — 전체 코어: ${FULL}s / 1코어 고정: ${ONE}s"

TASKSET_OK=$("$PY" -c "print('yes' if $ONE > $FULL * 1.3 else 'no')")
if [ "$TASKSET_OK" = "yes" ]; then
	echo "  → 코어 제한이 실제로 적용된다. 1 vCPU 측정을 진행한다."
else
	echo "  → ⚠ 코어 제한이 속도에 반영되지 않는다. 1 vCPU 측정은 신뢰할 수 없어 건너뛴다."
	echo "    (이 경우 1 vCPU 수치가 필요하면 c7g.medium을 직접 띄워 측정할 것)"
fi

# ---------------------------------------------------------------- 6. 측정
log "6/6 벤치마크"
mkdir -p results
TAG="$(uname -m)-$(nproc)core"

for f in normal crowd plates hires; do
	echo "  [$f] 전체 코어..."
	"$PY" bench.py "samples/$f.mp4" --device cpu --out "/tmp/out-$f.mp4" \
		> "results/ec2-${TAG}-$f.json" 2>/dev/null

	if [ "$TASKSET_OK" = "yes" ]; then
		echo "  [$f] 1 vCPU..."
		OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
			"$PY" bench.py "samples/$f.mp4" --device cpu --out "/tmp/out-$f-1c.mp4" \
			> "results/ec2-${TAG}-1vcpu-$f.json" 2>/dev/null
	fi
done

# ---------------------------------------------------------------- 결과
log "결과 요약"
"$PY" - <<'EOF'
import json, glob, os

rows = []
for path in sorted(glob.glob('results/ec2-*.json')):
	if not os.path.getsize(path):
		continue
	d = json.load(open(path))
	rows.append((
		os.path.basename(path).replace('ec2-', '').replace('.json', ''),
		d['resolution'], d['frames'], d['wall_sec'], d['realtime_factor'],
		d['ms_per_frame_inference'], d['peak_memory_mb'],
	))

w = max((len(r[0]) for r in rows), default=10)
print(f"{'측정':<{w}}  {'해상도':>10} {'프레임':>6} {'처리(s)':>9} {'실시간배수':>10} {'ms/frame':>9} {'메모리MB':>9}")
for r in rows:
	print(f"{r[0]:<{w}}  {r[1]:>10} {r[2]:>6} {r[3]:>9} {r[4]:>10} {r[5]:>9} {r[6]:>9}")

LAMBDA_LIMIT = 15 * 60
print("\nLambda 15분 제한 대비:")
for r in rows:
	print(f"  {r[0]:<{w}}  {100 * r[3] / LAMBDA_LIMIT:5.1f}%" + ("  ⚠ 초과" if r[3] > LAMBDA_LIMIT else ""))
EOF

echo -e "\n원시 결과: $WORKDIR/results/ec2-*.json"
echo "로컬로 가져오기:  scp -r ubuntu@<IP>:$WORKDIR/results/ec2-\\*.json ./results/"
