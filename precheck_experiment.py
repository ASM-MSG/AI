"""MSG-284 — 무의미 영상 프리체크: 판정 지표·임계값 실측.

암흑·렌즈 가림 영상을 추론 전에 걸러 건당 3분을 아끼는 게 목적이다(PRD: docs/prd/MSG-284-prd.md).
**이 실험이 답할 질문은 하나다 — 렌즈 가림과 야간 촬영을 무엇으로 가르는가.**

평균 밝기 단독으로 가르면 밤 거리 영상이 통째로 오탐된다. 렌즈 가림은 화면 전체가 균일하게
어둡고, 야간 거리는 가로등·간판 때문에 **밝은 점이 존재하고 대비가 크다** — 이 차이를 잡는
지표를 찾는 게 이 스크립트다. 여러 후보 지표를 한꺼번에 재서 분리 마진이 가장 큰 것을 고른다.

오탐(정상 영상 탈락)이 사고이고 미탐은 3분 낭비로 끝난다 — 블러 요구사항과 방향이 정반대라
**임계값은 PASS 그룹 최솟값보다 확실히 아래**로 잡는다.

    python precheck_experiment.py --smoke              # 합성 영상으로 지표 계산 검증
    python precheck_experiment.py                      # 전체 샘플
    python precheck_experiment.py samples/night-drive.mp4 --frames 30

디바이스 무관(순수 픽셀 통계, 추론 없음)이라 macOS 측정도 유효하다.

출력:
    results/MSG-284-precheck.json                      # 지표 원자료 (커밋 가능)
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

# 지표·임계값 정본은 bench.py다 — 실험이 검증하는 대상이 곧 배포되는 코드여야
# results/MSG-284-report.md가 근거로 유효하다 (MSG-284).
from bench import BRIGHT_LEVEL, PRECHECK_STD_THRESHOLD, frame_metrics, make_smoke_video, video_metrics

# 영상 분류. **PASS는 반드시 통과해야 하고 FAIL은 반드시 탈락해야 한다** — 임계값은 이 둘 사이.
# DIM은 crowd를 선형 감쇠한 저노출 스펙트럼으로, 경계가 어디쯤인지 연속적으로 보기 위한 참고군이다
# (판정 정답이 없다 — "애매하면 통과"가 정책이라 어느 쪽이든 오답은 아니다).
PASS_VIDEOS = [
	"samples/normal.mp4", "samples/crowd.mp4", "samples/plates.mp4", "samples/hires.mp4",
	"samples/kr-faces.mp4", "samples/kr-plates.mp4",
	"samples/night-street.mp4",  # 홍대 밤거리 — 어둡지만 얼굴·번호판이 있어 블러가 필요하다
	"samples/night-drive.mp4",   # 야간 주행 — 터널·고가, 조명이 띄엄띄엄
	"samples/night-dark.mp4",    # 한산한 야간 주택가 — 가로등만 있고 화면 대부분이 어둡다.
	                             # **임계값을 정하는 건 이 영상이다** — 통과해야 하는 것 중 가장 어둡다
]
FAIL_VIDEOS = ["samples/dark-covered.mp4"]
DIM_VIDEOS = ["samples/dim-50.mp4", "samples/dim-25.mp4", "samples/dim-10.mp4", "samples/dim-05.mp4"]

# 리포트 표(results/MSG-284-report.md)가 20장 측정이라 `python precheck_experiment.py`로
# 그대로 재현되도록 기본값을 유지한다. 배포 파이프라인은 10장(bench.PRECHECK_FRAMES)이지만
# std가 20/10/5장에서 ±0.2% 이내라 판정은 동일하다.
FRAMES = 20


def separation(rows, key):
	"""지표 하나가 PASS와 FAIL을 얼마나 벌려 놓는지. 마진이 클수록 임계값이 안전하다."""
	lo = min(r[key] for r in rows if r["group"] == "PASS")
	hi = max(r[key] for r in rows if r["group"] == "FAIL")
	# 모든 지표가 "낮을수록 탈락" 방향이다 — PASS 최솟값이 FAIL 최댓값보다 커야 가른다.
	# FAIL이 0이면 나눗셈이 아니라 **완전 분리**다 — None으로 두면 "못 가름"으로 오독된다.
	margin = float("inf") if hi == 0 else round(lo / hi, 2)
	return {"pass_min": lo, "fail_max": hi, "margin_x": margin if margin != float("inf") else "inf"}


def run(videos_by_group, n_frames):
	rows = []
	for group, videos in videos_by_group.items():
		for v in videos:
			if not Path(v).exists():
				print(f"  건너뜀(없음): {v}")
				continue
			print(f"[{group}] {v} ...", flush=True)
			m = video_metrics(v, n_frames)
			verdict = "통과" if m["std"] >= PRECHECK_STD_THRESHOLD else "탈락"
			rows.append({"video": Path(v).name, "group": group, **m, "verdict": verdict})

	metrics = [k for k in rows[0] if k not in ("video", "group", "verdict")]
	seps = {}
	if any(r["group"] == "PASS" for r in rows) and any(r["group"] == "FAIL" for r in rows):
		seps = {k: separation(rows, k) for k in metrics}
	return {
		"frames_per_video": n_frames,
		"bright_level": BRIGHT_LEVEL,
		"std_threshold": PRECHECK_STD_THRESHOLD,
		"note": "영상 지표는 프레임별 값의 중앙값. 전부 '낮을수록 탈락' 방향이다. "
			"DIM은 crowd 선형 감쇠본으로 판정 정답이 없는 참고군.",
		"rows": rows,
		"separation": seps,
	}


def print_table(report):
	metrics = [k for k in report["rows"][0] if k not in ("video", "group", "verdict")]
	head = f"{'video':<22}{'group':<7}" + "".join(f"{m:>11}" for m in metrics) + f"{'판정':>8}"
	print("\n" + head + "\n" + "-" * (len(head) + 2))
	for r in sorted(report["rows"], key=lambda r: (r["group"], -r["mean"])):
		# 판정이 그룹과 어긋나면 표시한다 — PASS인데 탈락이면 오탐(사고), FAIL인데 통과면 미탐
		flag = ""
		if r["group"] == "PASS" and r["verdict"] == "탈락":
			flag = "  ← 오탐!"
		elif r["group"] == "FAIL" and r["verdict"] == "통과":
			flag = "  ← 미탐"
		print(f"{r['video']:<22}{r['group']:<7}" + "".join(f"{r[m]:>11.2f}" for m in metrics)
			+ f"{r['verdict']:>7}{flag}")

	if not report["separation"]:
		return
	def order(kv):
		return float("inf") if kv[1]["margin_x"] == "inf" else kv[1]["margin_x"]

	print(f"\n{'지표':<12}{'PASS 최소':>12}{'FAIL 최대':>12}{'마진':>10}")
	print("-" * 46)
	for m, s in sorted(report["separation"].items(), key=order, reverse=True):
		mx = "완전분리" if s["margin_x"] == "inf" else f"{s['margin_x']:.2f}x"
		verdict = "" if order((m, s)) > 1 else "  ← 못 가름"
		print(f"{m:<12}{s['pass_min']:>12.2f}{s['fail_max']:>12.2f}{mx:>10}{verdict}")


def smoke():
	"""지표 계산이 극단 입력에서 죽지 않고 방향이 맞는지."""
	black = np.zeros((120, 160, 3), dtype=np.uint8)
	white = np.full((120, 160, 3), 255, dtype=np.uint8)
	bm, wm = frame_metrics(black), frame_metrics(white)

	assert bm["mean"] == 0 and wm["mean"] == 255, "평균 밝기 오계산"
	assert bm["bright_pct"] == 0 and wm["bright_pct"] == 100, "밝은 픽셀 비율 오계산"
	# 단색은 히스토그램이 한 칸에 몰리므로 엔트로피 0 — 로그 0 처리가 깨지면 nan이 난다
	assert bm["entropy"] == 0 and wm["entropy"] == 0, "단색 엔트로피는 0이어야 한다"

	# 어두워도 밝은 점이 있으면 bright_pct가 살아난다 = 야간 거리의 가로등.
	# **이게 이 실험의 가설이다** — mean만으로는 렌즈 가림과 구분되지 않는다.
	lamp = np.zeros((120, 160, 3), dtype=np.uint8)
	lamp[55:65, 75:85] = 255
	lm = frame_metrics(lamp)
	assert lm["bright_pct"] > 0, "밝은 점이 있는데 bright_pct가 0"
	assert lm["mean"] < 10, "가로등 근사가 어둡지 않다 — 테스트 설계 오류"

	# 채택 임계값이 극단 입력에서 방향을 지키는지 — 단색은 탈락, 명암이 있으면 통과
	assert bm["std"] < PRECHECK_STD_THRESHOLD and wm["std"] < PRECHECK_STD_THRESHOLD, \
		"단색은 대비가 0이라 탈락이어야 한다"

	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "in.mp4"
		make_smoke_video(src, seconds=3)
		report = run({"PASS": [str(src)]}, 5)
		assert report["rows"][0]["mean"] > 0, "합성 영상 지표 계산 실패"
		assert report["rows"][0]["verdict"] == "통과", "컬러바(testsrc)는 대비가 커서 통과여야 한다"
		assert not report["separation"], "FAIL 그룹이 없으면 분리 지표도 없어야 한다"
		print_table(report)
	print("\nsmoke OK")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("videos", nargs="*", help="지정하면 그 영상만 (그룹은 PASS로 간주)")
	ap.add_argument("--frames", type=int, default=FRAMES, help="영상당 샘플 프레임 수")
	ap.add_argument("--out", default="results/MSG-284-precheck.json")
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()

	if args.smoke:
		smoke()
		return

	groups = ({"PASS": args.videos} if args.videos
		else {"PASS": PASS_VIDEOS, "FAIL": FAIL_VIDEOS, "DIM": DIM_VIDEOS})
	report = run(groups, args.frames)
	Path(args.out).parent.mkdir(exist_ok=True)
	Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
	print_table(report)
	print(f"\n수치: {args.out}")


if __name__ == "__main__":
	main()
