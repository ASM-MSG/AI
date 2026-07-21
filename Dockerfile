# dev EC2 배포용 (MSG-143). 모델 가중치는 첫 잡 처리 때 HF에서 받는다 —
# 재시작마다 다시 받지 않도록 /root/.cache/huggingface를 볼륨으로 마운트할 것.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libgl1 libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bench.py server.py ./

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
