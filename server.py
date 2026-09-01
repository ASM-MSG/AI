"""MSG-161 — AI Highlight-Blur FastAPI 서버.

MSG-143 ADR: 상시 서버, 비동기 처리(1080p 30초에 3~4분), 파이프라인 첫 단계는
1080p 30fps 다운스케일(필수 전제). BE(Spring Boot)는 POST로 영상을 넘기고
GET /jobs/{id}를 폴링해 processing_status를 갱신한다. 계약은 README "API" 절.

    uvicorn server:app --host 0.0.0.0 --port 8000
    python server.py --smoke        # 합성 영상으로 API 왕복 검증

    AI_WORKERS=1                    # 허용값 1·2. 채택값은 results/MSG-339-report.md에서 확정
    ROUTE_AI_ENABLED=1              # MSG-458 경로 추천(/route/*) 활성 — 노브 4종 상세는 route_ai.py 상단

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
import route_ai

DEVICE = os.environ.get("DEVICE", "cpu")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "jobs"))
AI_WORKERS = int(os.environ.get("AI_WORKERS", "1"))
if AI_WORKERS not in (1, 2):
	raise ValueError("AI_WORKERS는 1, 2만 허용")
route_ai.validate_env()  # ROUTE_AI_ENABLED=1인데 키 없으면 기동 실패 (MSG-458 D-2, AI_WORKERS 검증 선례)

# ponytail: 인메모리 잡 저장소라 재시작하면 진행 중 잡이 유실된다.
# 실제 유실이 생기면 SQLite로 올린다.
jobs = {}
job_queue = queue.Queue()


def ffmpeg(*args):
	subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def downscale(src, dst):
	"""긴 변 1920 초과면 축소, 30fps 초과면 감쇠. 분석에 쓸 경로를 반환한다 (MSG-367 D-1).

	필터가 하나도 안 걸리면(BE가 이미 720p로 정규화한 입력) ffmpeg를 돌리지 않고 src를
	그대로 반환한다 — 무필터 재인코딩은 세대 손실만 보태는 버려질 패스였다. 필터를 태우면
	-an으로 오디오를 버린다 — 다운스케일본은 프레임 분석에만 쓰이고 최종 오디오는 항상
	원본에서 가져오므로(bench.run audio_src) 여기서의 aac 인코딩은 버려질 결과물이다.
	"""
	cap = cv2.VideoCapture(str(src))
	if not cap.isOpened():
		raise ValueError(f"영상을 열 수 없음: {src}")
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	cap.release()

	filters = []
	if max(w, h) > 1920:
		# 세로 영상(2160×3840)도 긴 변 기준으로 줄인다 → 1080×1920
		filters += ["-vf", "scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2"]
	if fps > 30:
		# -r 30은 영상 길이를 바꾸지 않으므로 원본 오디오와의 싱크는 유지된다 (docs/MSG-367.md D-1)
		filters += ["-r", "30"]
	if not filters:
		return src
	ffmpeg("-i", str(src), *filters, "-an", "-c:v", "libx264", "-preset", "veryfast", str(dst))
	return dst


def process(src, out_path, job):
	"""다운스케일(초과분만) → 프리체크 → 블러+하이라이트+단일 인코딩(bench.run) (MSG-367).

	bench.run의 출력이 곧 최종 재생본이다 — 별도 병합 패스와 mp4v 중간 파일이 없다.
	"""
	with tempfile.TemporaryDirectory() as tmp:
		analysis = downscale(src, Path(tmp) / "scaled.mp4")
		# MSG-284: 판정은 다운스케일 **다음**이다. 다운스케일은 전체 처리에서 차지하는 비중이 작아
		# 앞에서 걸러도 아끼는 게 없고, 4K 원본 판정은 프레임당 비용이 다운스케일본의 ~6배다.
		# 게다가 1080p로 정규화하면 분리 마진이 11.5배 → 14.2배로 넓어진다 (results/MSG-284-report.md)
		check = bench.precheck(analysis)
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
		# 오디오는 항상 잡 원본에서 가져온다 — 다운스케일을 탔으면 분석 입력에 오디오가 없다 (MSG-367 D-1·D-3)
		report = bench.run(analysis, DEVICE, out_path, audio_src=src)
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


def probe_duration(path):
	"""cv2 메타로 초 단위 길이. 못 열면 ValueError — /highlights 의 422 근거."""
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		raise ValueError(f"영상을 열 수 없음: {path}")
	frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	cap.release()
	return frames / fps if fps else 0.0


@app.post("/highlights")
def analyze_highlights(file: UploadFile):
	"""MSG-353 — 업로드 확정 전 선분석. 블러 없이 하이라이트만 동기 계산한다 (BE MSG-351이 소비).

	YOLO 모델이 필요 없어 워커 큐를 우회한다 — 블러 잡이 3분씩 돌아도 여기는 수 초다 (스펙 D-1).
	다운스케일 재인코딩도 없다 — 디코딩은 어차피 생략 불가라 재인코딩이 오히려 비싸고,
	감지 프레임 축소는 PySceneDetect가 내부에서 한다 (스펙 D-2). 잡 상태를 만들지 않으므로
	실패해도 서버에 아무것도 남지 않고, 원본 3분 상한 검증은 BE 몫이다 (PRD FR-8).
	"""
	with tempfile.TemporaryDirectory() as tmp:
		src = Path(tmp) / "src.mp4"
		with src.open("wb") as f:
			shutil.copyfileobj(file.file, f)
		try:
			duration = probe_duration(src)
			# 5초 미만 경계는 detect_highlights 가 빈 배열로 처리한다 (스펙 D-4)
			highlights = bench.detect_highlights(src, duration, bench.Stage())
		# bench 가 못 여는 입력에 SystemExit 을 던지므로 Exception 만으로는 부족하다 (worker 와 동일)
		except (Exception, SystemExit) as e:
			raise HTTPException(422, f"분석 불가: {str(e) or e.__class__.__name__}")
	return {"highlights": highlights}


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


# MSG-458: 경로 추천 언어 처리 — 동기, 워커 큐 우회 (D-1). 로직은 route_ai에 있고 여기는 상태 코드 매핑뿐이다


def route_call(pipeline, request):
	"""플래그 게이트 + route_ai outcome → 상태 코드 매핑 (MSG-458 계약 실패 표).

	detail에 사용자 문장·모델 출력 원문을 담지 않는다 (NFR-SEC-09).
	"""
	if not route_ai.is_enabled():
		raise HTTPException(503, "route AI disabled")
	try:
		return pipeline(request)
	except route_ai.RouteAiError as e:
		if e.outcome == "timeout":
			raise HTTPException(504, "모델 응답 시간 초과")
		raise HTTPException(502, "해석 실패")  # model_error·shape_reject — BE는 전부 실패로 받는다 (FR-ROUTE-08)


@app.post("/route/parse")
def route_parse(request: route_ai.ParseRequest):
	return route_call(route_ai.parse_route, request)


@app.post("/route/explain")
def route_explain(request: route_ai.ExplainRequest):
	return route_call(route_ai.explain_route, request)


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
		# 4K 60fps + 사인톤으로 만들어 다운스케일 경로와 원본 오디오 복원(MSG-367 D-3)까지 태운다
		bench.make_smoke_video(src, seconds=6, fps=60, size=(3840, 2160), audio_sec=6)

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
		# MSG-367: 재생본에 mp4v 경유 흔적이 없고, 다운스케일본이 -an이어도 오디오는 원본에서 복원된다
		assert bench.ffprobe_codec(out) == "h264", f"재생본 코덱이 h264가 아니다 (MSG-367): {bench.ffprobe_codec(out)}"
		assert bench.ffprobe_codec(out, "a:0") == "aac", \
			f"원본 오디오가 재생본에 없다 (MSG-367): {bench.ffprobe_codec(out, 'a:0')!r}"
		print(f"결과: {w}x{h} @ {fps:.0f}fps, highlights={job['highlights']}, precheck={job['precheck']}")

		# MSG-367 D-1: BE 정규화 경로(긴 변 ≤ 1920 · ≤ 30fps)는 재인코딩 없이 원본을 그대로 분석한다
		p720 = Path(tmp) / "p720.mp4"
		bench.make_smoke_video(p720, seconds=6, fps=30, size=(1280, 720))
		bypass_dst = Path(tmp) / "bypass-scaled.mp4"
		assert downscale(p720, bypass_dst) == p720, "무필터 입력인데 다운스케일 바이패스가 안 됐다 (MSG-367 D-1)"
		assert not bypass_dst.exists(), "바이패스인데 재인코딩 산출물이 생겼다 (MSG-367 D-1)"
		bypass_job = wait_done(upload(p720))
		assert bypass_job["status"] == "DONE", f"바이패스 잡 실패 (MSG-367): {bypass_job}"
		bypass_out = Path(tmp) / "bypass-out.mp4"
		bypass_out.write_bytes(client.get(f"/jobs/{bypass_job['job_id']}/video").content)
		assert bench.ffprobe_codec(bypass_out) == "h264", \
			f"바이패스 재생본이 h264가 아니다 (MSG-367): {bench.ffprobe_codec(bypass_out)}"
		cap = cv2.VideoCapture(str(bypass_out))
		bw, bh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		cap.release()
		assert (bw, bh) == (1280, 720), f"바이패스인데 해상도가 변했다 (MSG-367): {bw}x{bh}"
		print(f"바이패스 경로: {bw}x{bh}, highlights={bypass_job['highlights']}")

		# MSG-353: 선분석 전용 경로 — 큐를 안 거치는 동기 200. 4K 입력도 재인코딩 없이 직분석한다
		with src.open("rb") as f:
			r = client.post("/highlights", files={"file": ("in.mp4", f, "video/mp4")})
		assert r.status_code == 200, r.text
		pre = r.json()["highlights"]
		assert 1 <= len(pre) <= 3, f"6초 영상 선분석 구간 수 위반 (MSG-353): {pre}"
		for start, end in pre:
			assert end - start >= 4.99, f"5초 미만 구간 (MSG-353): {pre}"

		short = Path(tmp) / "short.mp4"
		bench.make_smoke_video(short, seconds=3)
		with short.open("rb") as f:
			r = client.post("/highlights", files={"file": ("short.mp4", f, "video/mp4")})
		assert r.status_code == 200 and r.json() == {"highlights": []}, \
			f"5초 미만은 빈 배열이어야 한다 (MSG-353): {r.text}"

		r = client.post("/highlights", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
		assert r.status_code == 422, f"열 수 없는 입력은 422 (MSG-353): {r.status_code}"
		print(f"선분석 경로: highlights={pre}")

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
		# MSG-458: 경로 추천 엔드포인트. 플래그 off(기본)는 두 경로 다 명시적 503 (PRD 운영 비기능)
		import httpx

		parse_body = {"text": "부산역 내려서 해운대에서 밥 먹고 축제도 보고 싶어",
			"viewport": {"min_lat": 35.05, "min_lng": 128.95, "max_lat": 35.25, "max_lng": 129.20}}
		explain_body = {"points": [
			{"name": "해운대 빛축제", "kind": "mission_festival", "facts": ["2026-08-01~08-31 진행 중"]},
			{"name": "광안리 해변", "kind": "place", "facts": ["이전 지점에서 1.2km"]},
		]}
		# 환경 플래그 상태를 전제하지 않는다 — EC2 실측은 ROUTE_AI_ENABLED=1로 돌린다 (Codex 2R).
		# 원래 상태를 저장하고 off→on을 강제해 두 분기를 다 태운 뒤 finally로 복원한다
		saved_enabled, real_call_model = route_ai.ROUTE_AI_ENABLED, route_ai.call_model

		def route_stub(raw):
			def _stub(system, user, timeout, response_format):
				if isinstance(raw, Exception):
					raise raw
				return raw
			route_ai.call_model = _stub

		try:
			route_ai.ROUTE_AI_ENABLED = False  # off 강제 — 두 엔드포인트 다 명시적 503
			for path, body in (("/route/parse", parse_body), ("/route/explain", explain_body)):
				r = client.post(path, json=body)
				assert r.status_code == 503, f"플래그 off인데 {path}가 503이 아니다: {r.status_code}"

			# on 강제 + call_model 스텁 왕복 — is_enabled() 요청 시점 조회(route_ai 전역 재대입)를 실제로 태운다
			route_ai.ROUTE_AI_ENABLED = True
			route_stub('{"region": "해운대", "period": {"start": "2026-08-22", "end": "2026-08-23"},'
				' "interests": ["맛집", "축제"], "preferred_order": ["부산역", "해운대 식사", "축제"],'
				' "related": true}')
			r = client.post("/route/parse", json=parse_body)
			assert r.status_code == 200, f"parse 왕복 실패: {r.status_code} {r.text}"
			assert r.json() == {"region": "해운대", "period": {"start": "2026-08-22", "end": "2026-08-23"},
				"interests": ["맛집", "축제"], "preferred_order": ["부산역", "해운대 식사", "축제"],
				"related": True}, \
				f"parse 200 형태 위반: {r.json()}"

			# 모델 내부 스키마는 index 에코 (Codex 4R) — BE로 나가는 응답 형태는 아래 어서션 그대로 불변이다
			route_stub('{"reasons": [{"index": 0, "reason": "8월 말까지 열리는 빛축제입니다."},'
				' {"index": 1, "reason": "식사 후 걷기 좋은 바다 지점입니다."}]}')
			r = client.post("/route/explain", json=explain_body)
			assert r.status_code == 200, f"explain 왕복 실패: {r.status_code} {r.text}"
			reasons = r.json()["reasons"]
			assert len(reasons) == 2 and reasons[0].startswith("8월"), f"explain 개수·순서 위반: {reasons}"
			assert "summary" not in r.json(), f"text 없는 explain 응답에 summary가 실렸다 (MSG-540 계약: 부재): {r.json()}"

			# MSG-540: text 동봉 왕복 — summary가 반드시 실리고, 구계약 출력(summary 없음)은 502다
			explain_text_body = {**explain_body, "text": parse_body["text"]}
			route_stub('{"reasons": [{"index": 0, "reason": "8월 말까지 열리는 빛축제입니다."},'
				' {"index": 1, "reason": "식사 후 걷기 좋은 바다 지점입니다."}],'
				' "summary": "축제와 식사를 말한 문장이라 해운대 축제와 바다 지점으로 묶었습니다."}')
			r = client.post("/route/explain", json=explain_text_body)
			assert r.status_code == 200, f"text 동봉 explain 왕복 실패: {r.status_code} {r.text}"
			assert r.json()["summary"].startswith("축제와"), f"summary가 훼손됐다: {r.json()}"
			route_stub('{"reasons": [{"index": 0, "reason": "8월 말까지 열리는 빛축제입니다."},'
				' {"index": 1, "reason": "식사 후 걷기 좋은 바다 지점입니다."}]}')
			r = client.post("/route/explain", json=explain_text_body)
			assert r.status_code == 502, f"text 동봉인데 summary 부재가 502가 아니다: {r.status_code}"

			route_stub('{"region": "해운대", "places": ["가짜 축제"]}')  # 미정의 필드 — 형태 위반 (FR-ROUTE-08)
			r = client.post("/route/parse", json=parse_body)
			assert r.status_code == 502, f"형태 위반이 502가 아니다: {r.status_code}"
			assert "해운대" not in r.text and "places" not in r.text, \
				f"502 detail에 모델 출력 원문이 샜다 (NFR-SEC-09): {r.text}"

			route_stub(httpx.TimeoutException("모의 타임아웃"))
			r = client.post("/route/parse", json=parse_body)
			assert r.status_code == 504, f"모델 타임아웃이 504가 아니다: {r.status_code}"

			r = client.post("/route/parse", json={"viewport": parse_body["viewport"]})  # text 결손
			assert r.status_code == 422, f"text 결손이 422가 아니다: {r.status_code}"
		finally:
			route_ai.ROUTE_AI_ENABLED, route_ai.call_model = saved_enabled, real_call_model
		assert (route_ai.ROUTE_AI_ENABLED, route_ai.call_model) == (saved_enabled, real_call_model), \
			"스모크가 route_ai 전역을 복원하지 않았다"
		# D-1: 동기 경로는 잡 상태를 만들지도 바꾸지도 않는다
		assert all(client.get(f"/jobs/{job_id}").json()["status"] == "DONE" for job_id in job_ids), \
			"route 호출이 기존 잡 상태를 바꿨다"
		print("route 경로: 503/200/502/504/422 OK")

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
