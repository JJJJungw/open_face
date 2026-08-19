#!/usr/bin/env bash
# face-anonymizer EC2 부트스트랩 + 스모크 테스트 (Ubuntu)
set -euo pipefail

# 저장소 자리는 이 파일 위치에서 찾는다. $HOME/face-anonymizer 로 박아 두면
# 다른 이름이나 다른 계정으로 클론한 사람은 첫 줄에서 cd 실패로 끝난다.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
VENV="${FA_VENV:-$REPO/.venv}"
cd "$REPO"
say(){ printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
warn(){ printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }

say "0/7 환경 확인"
echo "repo : $REPO"
. /etc/os-release 2>/dev/null || true
echo "os   : ${PRETTY_NAME:-unknown}"
echo "py   : $(python3 -V 2>&1)"
echo "cpu  : $(nproc) core / mem $(free -h | awk '/^Mem:/{print $2}')"
echo "disk : $(df -h --output=avail / | tail -1 | tr -d ' ') free on /"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  GPU=1; nvidia-smi -L
else
  GPU=0; echo "gpu  : 없음 → CPU 추론 (느림)"
fi

say "1/7 시스템 패키지 (ffmpeg = 오디오 합성, libgl = opencv 런타임)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip git ffmpeg
for p in libgl1 libgl1-mesa-glx libglib2.0-0t64 libglib2.0-0; do
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$p" >/dev/null 2>&1 || true
done

say "2/7 venv + torch"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -qU pip wheel
if [ "$GPU" = 1 ]; then
  "$VENV/bin/pip" install -q torch torchvision
else
  "$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 넷을 다 깐다. 예전에는 base + pytest 만 깔았는데, 그러면 fastapi·httpx·celery
# 가 없어 테스트 402개 중 247개가 importorskip 으로 조용히 빠진다 — 초록색
# "통과" 를 보면서 정작 이 기계에서 띄울 서버는 한 줄도 검증하지 않는 상태였다.
# 그리고 uvicorn 이 안 깔려 스크립트가 끝나도 서버를 못 띄웠다.
#
# 운영 이미지는 이렇게 만들지 않는다 — 워커 컨테이너는 worker.txt 만, 서버는
# serve.txt 만 깐다. 여기는 **이 기계에서 전 구간이 도는지 보는 자리**다.
say "3/7 의존성 (base + serve + worker + dev)"
"$VENV/bin/pip" install -q \
  -r requirements/base.txt \
  -r requirements/serve.txt \
  -r requirements/worker.txt \
  -r requirements/dev.txt
"$VENV/bin/pip" install -q -e .
"$VENV/bin/python" - <<'PY'
import cv2, numpy, supervision, torch
print(f"cv2={cv2.__version__}  numpy={numpy.__version__}  "
      f"supervision={supervision.__version__}  torch={torch.__version__}  "
      f"cuda={torch.cuda.is_available()}")
PY

say "4/7 YOLO-FaceV2 리포 + 가중치"
"$VENV/bin/python" scripts/setup_weights.py

# -q 를 붙이지 않는다. pyproject 의 addopts 에 이미 있어서 여기서 또 주면 -qq 가
# 되고, 그러면 "402 passed" 요약 줄까지 사라져 점만 찍히고 끝난다.
say "5/7 단위 테스트 (가짜 검출기 — 누출 방지 검증)"
LOG="${TMPDIR:-/tmp}/fa-pytest.log"
if "$VENV/bin/python" -m pytest -rs > "$LOG" 2>&1; then
  tail -1 "$LOG"
else
  tail -40 "$LOG"; echo; echo "전체 로그: $LOG"; exit 1
fi

# **skip 은 통과가 아니다.** 의존성이 빠져서 안 돈 것과 이 기계에 AV1 인코더가
# 없어서 못 돈 것은 뜻이 전혀 다르다. 앞의 것은 우리가 방금 깔았어야 할 것이라
# 여기서 멈추고, 뒤의 것은 사실대로 적고 넘어간다.
if grep -q '^SKIPPED' "$LOG"; then
  if grep '^SKIPPED' "$LOG" | grep -q 'pip install'; then
    warn "의존성이 빠져 건너뛴 테스트가 있다 — 3/7 단계가 제대로 안 끝났다."
    grep '^SKIPPED' "$LOG" | grep 'pip install' | sort -u
    exit 1
  fi
  warn "환경 때문에 건너뛴 것들 (의존성 문제 아님):"
  grep '^SKIPPED' "$LOG" | sed 's/^/  /' | sort -u
fi

# 3/7 이 실제로 서버를 띄울 수 있는 상태를 만들었는지 본다. 임포트만으로도
# static/index.html 누락(패키지 데이터)까지 같이 잡힌다 — 그건 기동에서 터진다.
say "6/7 서버 기동 준비 확인"
"$VENV/bin/python" - <<'PY'
import uvicorn, fastapi, boto3                      # noqa: F401
from face_anonymizer.service.server import app
print(f"fastapi={fastapi.__version__}  uvicorn={uvicorn.__version__}  "
      f"boto3={boto3.__version__}  routes={len(app.routes)}")
PY

say "7/7 실모델 스모크 — 합성 영상 60프레임 전 구간 통과"
"$VENV/bin/python" - <<'PY'
import cv2, numpy as np
w, h, n = 640, 480, 60
vw = cv2.VideoWriter('/tmp/synth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
assert vw.isOpened(), "mp4v 인코더가 없다"
for i in range(n):
    f = np.full((h, w, 3), 40, np.uint8)
    f[:, :, 1] = (np.arange(w, dtype=np.uint8)[None, :] * 3) % 255
    cv2.circle(f, (60 + i * 8, 240), 50, (180, 170, 200), -1)
    vw.write(f)
vw.release(); print("합성 영상 생성 완료: /tmp/synth.mp4")
PY
time "$VENV/bin/face-anonymize" /tmp/synth.mp4 -o /tmp/synth_anon.mp4 --imgsz 640
echo "--- 출력 검증 ---"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate \
  -of default=nk=1:nw=1 /tmp/synth_anon.mp4

printf '\n\033[1;32m✅ 배선 정상. (합성 영상이라 검출 0건이 정상 — 진짜 얼굴 영상으로 다음 단계 진행)\033[0m\n'
printf '   서버: %s/bin/uvicorn face_anonymizer.service.server:app --host 127.0.0.1 --port 8000\n' "$VENV"
