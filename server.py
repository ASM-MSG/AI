"""MSG-161 — AI Highlight-Blur FastAPI 서버.

MSG-143 ADR: 상시 서버, 비동기 처리(1080p 30초에 3~4분), 파이프라인 첫 단계는
1080p 30fps 다운스케일(필수 전제). BE(Spring Boot)는 POST로 영상을 넘기고
GET /jobs/{id}를 폴링해 processing_status를 갱신한다. 계약은 README "API" 절.

    uvicorn server:app --host 0.0.0.0 --port 8000
    python server.py --smoke        # 합성 영상으로 API 왕복 검증

    AI_WORKERS=1                    # 허용값 1·2. 채택값은 results/MSG-339-report.md에서 확정

블러·하이라이트 로직은 bench.py를 그대로 쓴다 — 서버는 껍데기다.
"""

import argparse
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

import bench

DEVICE = os.environ.get("DEVICE", "cpu")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "jobs"))
AI_WORKERS = int(os.environ.get("AI_WORKERS", "1"))
if AI_WORKERS not in (1, 2):
	raise ValueError("AI_WORKERS는 1, 2만 허용")

# ponytail: 인메모리 잡 저장소라 재시작하면 진행 중 잡이 유실된다.
# 실제 유실이 생기면 SQLite로 올린다.
jobs = {}
job_queue = queue.Queue()


def ffmpeg(*args):
	subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def downscale(src, dst):
	"""긴 변 1920 초과면 축소, 30fps 초과면 감쇠. 미달이면 업스케일하지 않는다."""
	cap = cv2.VideoCapture(str(src))
	if not cap.isOpened():
		raise ValueError(f"영상을 열 수 없음: {src}")
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	cap.release()

	args = ["-i", str(src)]
	if max(w, h) > 1920:
		# 세로 영상(2160×3840)도 긴 변 기준으로 줄인다 → 1080×1920
		args += ["-vf", "scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2"]
	if fps > 30:
		args += ["-r", "30"]
	args += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(dst)]
	ffmpeg(*args)


def process(src, out_path, job):
	"""다운스케일 → 프리체크 → 블러+하이라이트(bench.run) → h264 재인코딩 + 원본 오디오 복원."""
	with tempfile.TemporaryDirectory() as tmp:
		scaled = Path(tmp) / "scaled.mp4"
		raw = Path(tmp) / "raw.mp4"  # bench.run 출력 — mp4v·무음이라 그대로 못 내보낸다
		downscale(src, scaled)
		# MSG-284: 판정은 다운스케일 **다음**이다. 다운스케일은 전체 처리에서 차지하는 비중이 작아
		# 앞에서 걸러도 아끼는 게 없고, 4K 원본 판정은 프레임당 비용이 다운스케일본의 ~6배다.
		# 게다가 1080p로 정규화하면 분리 마진이 11.5배 → 14.2배로 넓어진다 (results/MSG-284-report.md)
		check = bench.precheck(scaled)
		# MSG-284 FR-7: 판정 **직후** 남긴다 — 뒤로 미루면 정상 영상은 3분 뒤에나 찍히고,
		# 그 사이 추론이 실패하면 영영 안 남는다. 실패한 잡일수록 판정 수치가 궁금하다.
		# 잡 디렉터리 이름이 곧 job_id다 (JOBS_DIR/{job_id}/src.mp4)
		print(f"[precheck] job={src.parent.name} passed={check['passed']} metrics={check['metrics']}", flush=True)
		# 계약 필드도 판정 직후에 채운다 — 뒤로 미루면 추론이 실패했을 때 판정 결과가 영영 안 남아,
		# FAILED 잡이 "판정 전에 죽은 것"과 구분되지 않는다 (계약: null = 판정 전).
		# 진단 수치(metrics)는 응답에 안 싣는다 — 로그로만 본다
		job["precheck"] = {"passed": check["passed"], "reason": check["reason"]}
		if not check["passed"]:
			# 탈락 잡은 추론·하이라이트·재인코딩을 전부 건너뛴다.
			# out.mp4를 만들지 않는 것이 곧 원본 유출 차단이다 (PRD FR-6)
			return {"highlights": []}
		report = bench.run(scaled, DEVICE, raw)
		ffmpeg("-i", str(raw), "-i", str(scaled), "-map", "0:v", "-map", "1:a?",
			"-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(out_path))
	return report


def worker():
	while True:
		job_id = job_queue.get()
		job = jobs[job_id]
		job["status"] = "PROCESSING"
		try:
			# job["precheck"]는 process()가 판정 직후에 채운다 — 추론이 실패해도 판정 결과는 남는다
			report = process(JOBS_DIR / job_id / "src.mp4", JOBS_DIR / job_id / "out.mp4", job)
			job["highlights"] = report["highlights"]
			job["status"] = "DONE"   # 탈락도 DONE이다 — BE 매핑 변경 없이 배포 가능해야 한다 (FR-5)
		# bench.run이 못 여는 입력에 SystemExit을 던지므로 Exception만으로는 부족하다
		except (Exception, SystemExit) as e:
			job["error"] = str(e) or e.__class__.__name__
			job["status"] = "FAILED"


worker_threads = [threading.Thread(target=worker, daemon=True) for _ in range(AI_WORKERS)]
for thread in worker_threads:
	thread.start()

app = FastAPI(title="FillMap AI Highlight-Blur")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/jobs", status_code=202)
def create_job(file: UploadFile):
	job_id = uuid.uuid4().hex
	job_dir = JOBS_DIR / job_id
	job_dir.mkdir(parents=True)
	with (job_dir / "src.mp4").open("wb") as f:
		shutil.copyfileobj(file.file, f)
	# precheck는 highlights·error와 같은 "판정 전에는 null" 패턴 — 응답 스키마가 상태와 무관하게 일정하다
	jobs[job_id] = {"job_id": job_id, "status": "QUEUED", "highlights": None, "error": None, "precheck": None}
	job_queue.put(job_id)
	return {"job_id": job_id, "status": "QUEUED"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
	if job_id not in jobs:
		raise HTTPException(404, "없는 job")
	return jobs[job_id]


@app.get("/jobs/{job_id}/video")
def get_video(job_id: str):
	if job_id not in jobs:
		raise HTTPException(404, "없는 job")
	if jobs[job_id]["status"] != "DONE":
		raise HTTPException(409, f"아직 결과 없음: {jobs[job_id]['status']}")
	precheck = jobs[job_id]["precheck"]
	if precheck and not precheck["passed"]:
		# MSG-284: 탈락 잡도 DONE이라 위 검사만으로는 통과해버린다. 블러본이 없으니
		# 원본을 대신 내보내면 프라이버시 사고다 (PRD FR-6)
		raise HTTPException(409, f"프리체크 탈락: {precheck['reason']}")
	return FileResponse(JOBS_DIR / job_id / "out.mp4", media_type="video/mp4")


def smoke():
	"""합성 영상으로 API 왕복(업로드 → 폴링 → 결과 다운로드) 검증."""
	global JOBS_DIR
	from fastapi.testclient import TestClient

	with tempfile.TemporaryDirectory() as tmp:
		JOBS_DIR = Path(tmp) / "jobs"
		expected_workers = int(os.environ.get("AI_WORKERS", "1"))
		assert len(globals().get("worker_threads", [])) == expected_workers, "설정한 워커 수와 다르다"
		assert all(thread.is_alive() for thread in globals().get("worker_threads", [])), "종료된 워커가 있다"
		src = Path(tmp) / "in.mp4"
		# 4K 60fps로 만들어 다운스케일 경로까지 태운다
		bench.make_smoke_video(src, seconds=6, fps=60, size=(3840, 2160))

		client = TestClient(app)
		assert client.get("/health").json() == {"status": "ok"}

		def upload(path):
			with Path(path).open("rb") as f:
				r = client.post("/jobs", files={"file": (Path(path).name, f, "video/mp4")})
			assert r.status_code == 202, r.text
			return r.json()["job_id"]

		def wait_done(job_id):
			deadline = time.time() + 300
			while time.time() < deadline:
				job = client.get(f"/jobs/{job_id}").json()
				if job["status"] in ("DONE", "FAILED"):
					return job
				time.sleep(2)
			raise AssertionError(f"처리 시간 초과: {job_id}")

		job_ids = [upload(src), upload(src)]
		for job_id in job_ids:
			assert client.get(f"/jobs/{job_id}/video").status_code == 409, "완료 전엔 409여야 한다"

		done_jobs = [wait_done(job_id) for job_id in job_ids]
		videos = [client.get(f"/jobs/{job_id}/video") for job_id in job_ids]
		for job, video in zip(done_jobs, videos):
			assert job["status"] == "DONE", f"처리 실패: {job}"
			assert len(job["highlights"]) <= 3, "하이라이트는 최대 3구간 (MSG-141)"
			assert job["precheck"] == {"passed": True, "reason": None}, \
				f"정상 영상 프리체크 계약 위반: {job['precheck']}"
			assert video.status_code == 200 and len(video.content) > 0, "결과 영상이 비었다"

		job, video = done_jobs[0], videos[0]
		assert client.get("/jobs/없는아이디").status_code == 404

		out = Path(tmp) / "out.mp4"
		out.write_bytes(video.content)
		cap = cv2.VideoCapture(str(out))
		w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		cap.release()
		assert max(w, h) <= 1920, f"다운스케일 안 됨: {w}x{h}"
		assert fps <= 30.5, f"30fps 초과: {fps}"
		print(f"결과: {w}x{h} @ {fps:.0f}fps, highlights={job['highlights']}, precheck={job['precheck']}")

		# MSG-284 탈락 경로. 처리 시간 어서션은 넣지 않는다 — macOS 측정은 무효라
		# 성능 요건(탈락 잡 ≤ 정상 잡의 1/5)은 dev EC2 실측으로 판정한다 (CLAUDE.md "알아둘 함정")
		dark = Path(tmp) / "dark.mp4"
		bench.make_dark_video(dark, dark_sec=6)
		dark_id = upload(dark)
		dark_job = wait_done(dark_id)
		assert dark_job["status"] == "DONE", f"탈락 잡도 DONE이어야 한다 (FR-5): {dark_job}"
		assert dark_job["highlights"] == [], f"탈락 잡의 하이라이트는 빈 배열 (FR-5): {dark_job['highlights']}"
		assert dark_job["precheck"]["passed"] is False, f"암흑 영상이 통과했다: {dark_job['precheck']}"
		assert isinstance(dark_job["precheck"]["reason"], str), "탈락 사유가 문자열이 아니다 (FR-4)"
		assert client.get(f"/jobs/{dark_id}/video").status_code == 409, "탈락 잡 영상은 409여야 한다 (FR-6)"
		# 파일이 없다는 사실 자체가 원본 유출 차단이다 (FR-6)
		assert not (JOBS_DIR / dark_id / "out.mp4").exists(), "탈락 잡인데 out.mp4가 생성됐다"
		print(f"탈락 경로: precheck={dark_job['precheck']}")

		broken = Path(tmp) / "broken.mp4"
		broken.write_bytes(b"not a video")
		broken_id = upload(broken)
		assert client.get(f"/jobs/{broken_id}/video").status_code == 409, \
			"실패 전 손상 입력 영상은 409여야 한다"
		broken_job = wait_done(broken_id)
		assert broken_job["status"] == "FAILED", f"손상 입력이 FAILED가 아니다: {broken_job}"
		assert all(client.get(f"/jobs/{job_id}").json()["status"] == "DONE" for job_id in job_ids), \
			"손상 입력이 정상 작업 상태를 바꿨다"
		print("smoke OK")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--smoke", action="store_true", help="합성 영상으로 API 왕복 검증")
	args = ap.parse_args()
	if args.smoke:
		smoke()
	else:
		import uvicorn
		uvicorn.run(app, host="0.0.0.0", port=8000)
