# 비식별화 서비스 컨테이너.
#
# **GPU 베이스 이미지를 안 쓴다.** torch 의 cu121 휠이 CUDA 런타임을 통째로
# 들고 오므로, 호스트에 NVIDIA 드라이버와 nvidia-container-toolkit 만 있으면
# 된다. `nvidia/cuda` 위에 파이썬을 올리는 것보다 층이 하나 적고, 붙을 곳
# (rebornstudio worker-io)이 쓰는 베이스와 같아진다.
#
#   docker build -t face-anonymizer .
#   docker run --gpus all -p 8000:8000 -e FA_REMOTE_OPEN=1 face-anonymizer
#
# GPU 가 없어도 뜬다. 느릴 뿐이다.
FROM python:3.11-slim

# ffmpeg   오디오 합성·코덱 정규화 (core/ingest.py)
# libgl1·libglib2.0-0  opencv 가 링크한다. 없으면 임포트에서 죽는다
# git      검출기 리포를 클론한다 (scripts/setup_weights.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# **torch 를 먼저, 따로 넣는다.** 2GB 가 넘는 층이라 여기서 끊어 두면 우리 코드만
# 고쳤을 때 다시 안 받는다. CPU 로만 돌릴 거면 `--build-arg TORCH_INDEX=` 로
# 비워서 PyPI 기본 휠을 쓴다.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
RUN if [ -n "$TORCH_INDEX" ]; then \
      pip install --no-cache-dir --index-url "$TORCH_INDEX" torch torchvision; \
    else \
      pip install --no-cache-dir torch torchvision; \
    fi

# 나머지 의존성. torch 는 위에서 이미 채워졌으므로 여기서 다시 받지 않는다
# (base.txt 의 `torch>=1.13` 이 이미 만족된다).
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/serve.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# **가중치를 이미지에 굽는다.** 기동할 때 받으면 첫 영상이 40MB 다운로드를
# 기다리고, 바깥으로 못 나가는 망에서는 아예 못 뜬다. 빌드에 네트워크가 필요한
# 것이 그 대가다 — 막힌 곳에서는 `--build-arg BAKE_WEIGHTS=0` 으로 끄고,
# 런타임에 `weights/` 를 볼륨으로 물린다.
#
# 검출기 **리포**는 굽지 않을 수 없다. 체크포인트를 unpickle 하려면 그 리포의
# 모듈이 임포트돼야 해서, 없으면 모델 로드가 통째로 실패한다.
ARG BAKE_WEIGHTS=1
RUN python scripts/setup_weights.py $( [ "$BAKE_WEIGHTS" = "1" ] || echo --repo-only )

# 작업 디렉터리는 볼륨으로 뺀다. 컨테이너가 지워져도 진행 중이던 것의 흔적이
# 남고, 같은 폴더를 두 서버가 잡는 사고는 .server.lock 이 막는다.
ENV FA_JOBS_DIR=/data/jobs \
    FA_PRELOAD=1 \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8000

# 한 편이 분 단위다. 종료 신호를 받고 정리할 시간을 넉넉히 준다 —
# compose 쪽에서 stop_grace_period 로 맞춘다.
CMD ["uvicorn", "face_anonymizer.service.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
