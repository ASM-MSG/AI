"""MSG-161 — AI Highlight-Blur FastAPI 서버.

MSG-143 ADR: 상시 서버, 비동기 처리(1080p 30초에 3~4분), 파이프라인 첫 단계는
1080p 30fps 다운스케일(필수 전제). BE(Spring Boot)는 POST로 영상을 넘기고
GET /jobs/{id}를 폴링해 processing_status를 갱신한다. 계약은 README "API" 절.

    uvicorn server:app --host 0.0.0.0 --port 8000
    python server.py --smoke        # 합성 영상으로 API 왕복 검증

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

# ponytail: 인메모리 잡 저장소 + 단일 워커 스레드 — 재시작하면 진행 중 잡이 유실된다.
# 1 vCPU에 순차 처리가 전제(MSG-143)라 큐도 스레드도 하나면 된다.
# 유실이 실제 문제가 되면 SQLite, 병렬이 필요해지면 그때 프로세스 풀.
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


def process(src, out_path):
	"""다운스케일 → 블러+하이라이트(bench.run) → h264 재인코딩 + 원본 오디오 복원."""
	with tempfile.TemporaryDirectory() as tmp:
		scaled = Path(tmp) / "scaled.mp4"
		raw = Path(tmp) / "raw.mp4"  # bench.run 출력 — mp4v·무음이라 그대로 못 내보낸다
		downscale(src, scaled)
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
			report = process(JOBS_DIR / job_id / "src.mp4", JOBS_DIR / job_id / "out.mp4")
			job["highlights"] = report["highlights"]
			job["status"] = "DONE"
		# bench.run이 못 여는 입력에 SystemExit을 던지므로 Exception만으로는 부족하다
		except (Exception, SystemExit) as e:
			job["error"] = str(e) or e.__class__.__name__
			job["status"] = "FAILED"


threading.Thread(target=worker, daemon=True).start()

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
	jobs[job_id] = {"job_id": job_id, "status": "QUEUED", "highlights": None, "error": None}
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
	return FileResponse(JOBS_DIR / job_id / "out.mp4", media_type="video/mp4")


def smoke():
	"""합성 영상으로 API 왕복(업로드 → 폴링 → 결과 다운로드) 검증."""
	global JOBS_DIR
	from fastapi.testclient import TestClient

	with tempfile.TemporaryDirectory() as tmp:
		JOBS_DIR = Path(tmp) / "jobs"
		src = Path(tmp) / "in.mp4"
		# 4K 60fps로 만들어 다운스케일 경로까지 태운다
		bench.make_smoke_video(src, seconds=6, fps=60, size=(3840, 2160))

		client = TestClient(app)
		assert client.get("/health").json() == {"status": "ok"}

		with src.open("rb") as f:
			r = client.post("/jobs", files={"file": ("in.mp4", f, "video/mp4")})
		assert r.status_code == 202, r.text
		job_id = r.json()["job_id"]

		assert client.get(f"/jobs/{job_id}/video").status_code == 409, "완료 전엔 409여야 한다"

		deadline = time.time() + 300
		while time.time() < deadline:
			job = client.get(f"/jobs/{job_id}").json()
			if job["status"] in ("DONE", "FAILED"):
				break
			time.sleep(2)
		assert job["status"] == "DONE", f"처리 실패: {job}"
		assert len(job["highlights"]) <= 3, "하이라이트는 최대 3구간 (MSG-141)"

		video = client.get(f"/jobs/{job_id}/video")
		assert video.status_code == 200 and len(video.content) > 0, "결과 영상이 비었다"
		assert client.get("/jobs/없는아이디").status_code == 404

		out = Path(tmp) / "out.mp4"
		out.write_bytes(video.content)
		cap = cv2.VideoCapture(str(out))
		w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		cap.release()
		assert max(w, h) <= 1920, f"다운스케일 안 됨: {w}x{h}"
		assert fps <= 30.5, f"30fps 초과: {fps}"
		print(f"결과: {w}x{h} @ {fps:.0f}fps, highlights={job['highlights']}")
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
