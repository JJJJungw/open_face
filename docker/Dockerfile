# GPU 서빙용 이미지.
#   docker build -f docker/Dockerfile -t face-anonymizer .
#   docker run --gpus all -p 8000:8000 \
#       -v $PWD/weights:/app/weights -v $PWD/third_party:/app/third_party \
#       face-anonymizer
#
# 가중치와 YOLO-FaceV2 리포는 이미지에 굽지 않고 볼륨으로 넣는다.
# (이미지 크기 + 라이선스 + 갱신 편의)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FA_WORKDIR=/tmp/face-anon \
    FA_BATCH_SIZE=16 \
    FA_EAGER_LOAD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml README.md LICENSE setup_weights.py ./
COPY face_anonymizer ./face_anonymizer
RUN pip install --no-deps -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "face_anonymizer.server:app", "--host", "0.0.0.0", "--port", "8000"]
