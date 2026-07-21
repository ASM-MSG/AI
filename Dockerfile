# dev EC2 배포용 (MSG-143). 모델 가중치는 첫 잡 처리 때 HF에서 받는다 —
# 재시작마다 다시 받지 않도록 /root/.cache/huggingface를 볼륨으로 마운트할 것.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg libgl1 libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# x86_64는 기본 wheel이 CUDA를 끌고 와 2GB+ 받는다 — CPU 전용 인덱스로 먼저 설치.
# (arm64도 같은 인덱스에 CPU wheel이 있어 분기 불필요. ec2-bench.sh와 같은 해법)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
	&& pip install --no-cache-dir -r requirements.txt

COPY bench.py server.py ./

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
