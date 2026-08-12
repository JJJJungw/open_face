#!/usr/bin/env bash
# face-anonymizer EC2 부트스트랩 + 스모크 테스트 (Ubuntu)
set -euo pipefail
REPO="$HOME/face-anonymizer"; VENV="$REPO/.venv"; cd "$REPO"
say(){ printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

say "0/6 환경 확인"
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

say "1/6 시스템 패키지 (ffmpeg = 오디오 합성, libgl = opencv 런타임)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip git ffmpeg
for p in libgl1 libgl1-mesa-glx libglib2.0-0t64 libglib2.0-0; do
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$p" >/dev/null 2>&1 || true
done

say "2/6 venv + torch"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -qU pip wheel
if [ "$GPU" = 1 ]; then
  "$VENV/bin/pip" install -q torch torchvision
else
  "$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

say "3/6 나머지 의존성"
"$VENV/bin/pip" install -q -r requirements.txt pytest
"$VENV/bin/pip" install -q -e .
"$VENV/bin/python" - <<'PY'
import cv2, numpy, supervision, torch
print(f"cv2={cv2.__version__}  numpy={numpy.__version__}  "
      f"supervision={supervision.__version__}  torch={torch.__version__}  "
      f"cuda={torch.cuda.is_available()}")
PY

say "4/6 YOLO-FaceV2 리포 + 가중치"
"$VENV/bin/python" setup_weights.py

say "5/6 단위 테스트 (가짜 검출기 — 누출 방지 검증)"
"$VENV/bin/python" -m pytest -q

say "6/6 실모델 스모크 — 합성 영상 60프레임 전 구간 통과"
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
