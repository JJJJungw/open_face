#!/usr/bin/env python3
"""MSA 큐 왕복 한 바퀴 — 저쪽 없이 우리끼리 돌려 본다.

붙일 곳(RebornStudio)이 아직 없으므로, 저쪽이 할 일을 **이 스크립트가 흉내 낸다.**

    ① 서명된 URL 을 대신하는 로컬 HTTP 서버  (GET 원본 / PUT 산출)
    ② 잡을 큐에 넣는 발신자                  (저쪽 build_jobs)
    ③ 하트비트·완료를 받는 수신자            (저쪽 heartbeat / complete)

가운데에서 도는 것은 **진짜 우리 워커**다(`face_anonymizer.msa.celery_app`).
가짜로 바꾸는 것이 하나도 없다 — 실제 가중치로 실제 GPU 를 쓴다. 그래서 여기서
나오는 **처리 시간이 그대로 운영 숫자**이고, 저쪽 KEDA 의 "대기 몇 건당 워커
한 대" 를 정하는 근거가 된다.

사용::

    redis-server --daemonize yes                       # 없으면
    python tools/msa_smoke.py --input sample.mp4

    python tools/msa_smoke.py --input sample.mp4 --repeat 3   # 평균을 보려면

끝나면 결과 영상은 `--outdir`(기본 ./msa-smoke-out) 에 떨어진다.
"""

import argparse
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

TOKEN = "smoke-token-0001"           # 저쪽이 발급했다고 치는 펜싱 토큰


# ── ① 서명된 URL 흉내 ────────────────────────────────────────────────────────

class _Bucket(http.server.BaseHTTPRequestHandler):
    """GET 은 원본을, PUT 은 산출물을 받는다. presigned URL 자리에 들어간다."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        blob = self.server.blob
        self.send_response(200)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)
        print(f"  [스토리지] GET  {self.path} → {len(blob)/1e6:.1f} MB", flush=True)

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        dest = os.path.join(self.server.outdir, os.path.basename(self.path))
        with open(dest, "wb") as f:
            f.write(data)
        self.send_response(200)
        self.end_headers()
        print(f"  [스토리지] PUT  {self.path} ← {len(data)/1e6:.1f} MB → {dest}",
              flush=True)


def start_bucket(video, outdir, port):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Bucket)
    with open(video, "rb") as f:
        srv.blob = f.read()
    srv.outdir = outdir
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ── ③ 저쪽 api-gateway 흉내 (자식 프로세스로 뜬다) ───────────────────────────

FAKE_WORKER_SRC = '''
"""--fake-detector 전용 — 진짜 워커에 가짜 검출기만 끼운다.

가중치도 GPU 도 없는 곳에서 **배선만** 보고 싶을 때 쓴다. 큐·전송·하트비트·펜싱은
전부 진짜 경로를 탄다. 처리 시간은 의미 없다.
"""
from face_anonymizer.core.pipeline import VideoAnonymizer
from face_anonymizer.msa import celery_app as shell
from face_anonymizer.msa.celery_app import app          # celery -A 가 찾는 이름


class _Fake:
    def detect_batch(self, frames, imgsz=None, conf=None, iou=None):
        return [[] for _ in frames]


shell._anonymizer = VideoAnonymizer(detector=_Fake())
print("[smoke] 가짜 검출기 주입 — 배선만 봅니다", flush=True)
'''

OBSERVER_SRC = '''
"""저쪽 api-gateway 흉내 — 하트비트·완료를 받아 펜싱 토큰을 대조한다."""
import json, os
from celery import Celery

app = Celery("smoke-observer", broker=os.environ["FA_BROKER_URL"])
app.conf.update(accept_content=["json"], task_serializer="json",
                result_backend=None, task_ignore_result=True)
TOKEN = os.environ["SMOKE_TOKEN"]
REPORT = os.environ["SMOKE_REPORT"]

@app.task(name=os.environ["SMOKE_HEARTBEAT_TASK"], ignore_result=True)
def heartbeat(video_id=None, token=None, progress_s=None):
    ok = "리스 연장" if token == TOKEN else "★토큰 불일치 → 무시★"
    print(f"  [저쪽 API] 하트비트 {video_id} progress={progress_s}s → {ok}", flush=True)

@app.task(name=os.environ["SMOKE_COMPLETE_TASK"], ignore_result=True)
def complete(video_id=None, token=None, ok=None, **kw):
    verdict = "인정(done)" if token == TOKEN else "★토큰 불일치 → 버림★"
    print(f"  [저쪽 API] 완료보고 {video_id} ok={ok} → {verdict}", flush=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"video_id": video_id, "token": token, "ok": ok,
                            "accepted": token == TOKEN, **kw},
                           ensure_ascii=False) + "\\n")
'''


def spawn(argv, env, log):
    fh = open(log, "wb")
    p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=ROOT)
    p._log = log
    return p


def wait_ready(log, needle, proc, timeout=180):
    """워커가 브로커에 붙을 때까지 기다린다. 죽으면 로그를 보여 준다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(open(log, encoding="utf-8", errors="replace").read()[-3000:])
            raise SystemExit(f"× 워커가 뜨다 죽었습니다 — {log}")
        try:
            if needle in open(log, encoding="utf-8", errors="replace").read():
                return
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"× 워커가 {timeout}초 안에 준비되지 않았습니다 — {log}")


def main():
    ap = argparse.ArgumentParser(description="MSA 큐 왕복 한 바퀴")
    ap.add_argument("--input", required=True, help="테스트할 영상 한 편")
    ap.add_argument("--broker", default=os.environ.get("REDIS_URL")
                    or "redis://127.0.0.1:6379/0")
    ap.add_argument("--outdir", default="./msa-smoke-out")
    ap.add_argument("--port", type=int, default=9301)
    ap.add_argument("--repeat", type=int, default=1, help="몇 편 넣을까(평균용)")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--fake-detector", action="store_true",
                    help="가중치·GPU 없이 배선만 본다 (처리 시간은 의미 없음)")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if not os.path.exists(a.input):
        raise SystemExit(f"× 영상이 없습니다: {a.input}")
    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    report = os.path.join(outdir, "report.jsonl")
    if os.path.exists(report):
        os.remove(report)

    base = f"http://127.0.0.1:{a.port}"
    env = dict(os.environ)
    env.update(
        FA_BROKER_URL=a.broker,
        SMOKE_TOKEN=TOKEN,
        SMOKE_REPORT=report,
        SMOKE_HEARTBEAT_TASK=env.get("FA_MSA_HEARTBEAT_TASK",
                                     "worker_io.tasks.deidentify_heartbeat"),
        SMOKE_COMPLETE_TASK=env.get("FA_MSA_COMPLETE_TASK",
                                    "worker_io.tasks.deidentify_complete"),
        PYTHONPATH=ROOT + os.pathsep + logdir + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
    )
    queue = env.get("FA_MSA_QUEUE", "q.deidentify")
    task = env.get("FA_MSA_TASK", "worker_io.tasks.deidentify_one")

    from celery import Celery
    probe = Celery("smoke-probe", broker=a.broker)
    try:
        with probe.connection_for_write() as c:
            c.ensure_connection(max_retries=2)
    except Exception as e:                                  # noqa: BLE001
        raise SystemExit(f"× 브로커에 못 붙었습니다 ({a.broker}): {e}\n"
                         f"  redis-server --daemonize yes 로 띄우고 다시 해 주세요.")

    print(f"■ 브로커  {a.broker}")
    print(f"■ 큐      {queue}   태스크 {task}")
    print(f"■ 산출    {outdir}\n")

    with open(os.path.join(logdir, "smoke_observer.py"), "w", encoding="utf-8") as f:
        f.write(OBSERVER_SRC)
    worker_app = "face_anonymizer.msa.celery_app"
    if a.fake_detector:
        with open(os.path.join(logdir, "smoke_worker.py"), "w", encoding="utf-8") as f:
            f.write(FAKE_WORKER_SRC)
        worker_app = "smoke_worker"
        env["FA_MSA_PRELOAD"] = "0"
        print("  ⚠ 가짜 검출기 모드 — 배선만 봅니다. 처리 시간은 의미 없습니다.\n")

    srv = start_bucket(a.input, outdir, a.port)
    print(f"① 스토리지 흉내 준비 — {base}")

    obs = spawn([sys.executable, "-m", "celery", "-A", "smoke_observer",
                 "worker", "-Q", env.get("FA_MSA_CALLBACK_QUEUE", "default"),
                 "-c", "2", "-l", "info", "-n", "smoke-api@%h"],
                env, os.path.join(logdir, "observer.log"))
    wrk = spawn([sys.executable, "-m", "celery", "-A", worker_app, "worker",
                 "-Q", queue, "-c", "1", "--prefetch-multiplier", "1",
                 "-l", "info", "-n", "smoke-deid@%h"],
                env, os.path.join(logdir, "worker.log"))

    try:
        print("② 저쪽 API 흉내 기동…")
        wait_ready(obs._log, "ready.", obs)
        print("③ 우리 워커 기동 (모델 로딩 포함, 처음이면 오래 걸립니다)…")
        wait_ready(wrk._log, "ready.", wrk)
        print("   준비 완료\n")

        producer = Celery("smoke-producer", broker=a.broker)
        producer.conf.update(task_serializer="json", accept_content=["json"])
        t0 = time.time()
        for i in range(a.repeat):
            vid = f"smoke-{i:03d}"
            producer.send_task(task, queue=queue, args=[{
                "video_id": vid,
                "token": TOKEN,
                "input_url": f"{base}/in.mp4",
                "targets": [{
                    "label": "deid-720p", "height": 720,
                    "bitrate": "3500k", "max_bitrate": "4000k",
                    "method": "mosaic", "conf": 0.25,
                    "put_url": f"{base}/{vid}.mp4",
                    "content_type": "video/mp4"}],
                "heartbeat_every_s": 10,
            }])
            print(f"④ 잡 투입 {vid} — token={TOKEN}")
        print()

        deadline = time.time() + a.timeout
        seen = 0
        while seen < a.repeat and time.time() < deadline:
            if wrk.poll() is not None:
                print(open(wrk._log, encoding="utf-8", errors="replace").read()[-3000:])
                raise SystemExit("× 워커가 죽었습니다")
            if os.path.exists(report):
                seen = sum(1 for _ in open(report, encoding="utf-8"))
            time.sleep(0.5)
        wall = time.time() - t0

        # 마지막으로 던진 펜싱 확인 — 남의 토큰으로 온 보고는 버려져야 한다
        producer.send_task(env["SMOKE_COMPLETE_TASK"],
                           queue=env.get("FA_MSA_CALLBACK_QUEUE", "default"),
                           kwargs={"video_id": "smoke-stale", "token": "남의-토큰",
                                   "ok": True})
        time.sleep(3)

        rows = [json.loads(l) for l in open(report, encoding="utf-8")] \
            if os.path.exists(report) else []
        report_out(rows, wall, a, outdir)
    finally:
        for p in (wrk, obs):
            p.send_signal(signal.SIGTERM)
        time.sleep(2)
        for p in (wrk, obs):
            if p.poll() is None:
                p.kill()
        srv.shutdown()
        # 로그는 지우지 않는다. 실패했을 때 있는 것이 그것뿐이다.


def report_out(rows, wall, a, outdir):
    print("\n" + "=" * 62)
    done = [r for r in rows if r.get("ok") and r.get("accepted")]
    stale = [r for r in rows if not r.get("accepted")]
    fail = [r for r in rows if r.get("ok") is False]

    for r in rows:
        if not r.get("accepted"):
            continue                 # 펜싱에 걸린 보고는 아래에 따로 센다
        if r.get("ok"):
            t = (r.get("targets") or [{}])[0]
            print(f"  ✔ {r['video_id']}  {r.get('elapsed_s')}s  "
                  f"프레임 {t.get('frames')}  검출된 프레임 {t.get('detected_frames')}  "
                  f"실시간 대비 {t.get('realtime_factor')}x")
        else:
            print(f"  ✘ {r['video_id']}  [{r.get('stage')}] "
                  f"transient={r.get('transient')}  {r.get('error')}")
    if stale:
        print(f"  ⓘ 펜싱: 남의 토큰으로 온 보고 {len(stale)}건 → 버려짐 (정상)")

    if done:
        secs = [r["elapsed_s"] for r in done]
        avg = sum(secs) / len(secs)
        print("-" * 62)
        print(f"  성공 {len(done)}/{a.repeat}   실패 {len(fail)}")
        print(f"  한 편 평균 {avg:.1f}s   (최소 {min(secs):.1f} / 최대 {max(secs):.1f})")
        print(f"  전체 벽시계 {wall:.1f}s")
        print()
        print(f"  → KEDA listLength 근거: 워커 1대가 시간당 약 "
              f"{3600/avg:.0f}편을 처리합니다.")
    else:
        print("-" * 62)
        print("  완료된 건이 없습니다. logs/worker.log 를 봐 주세요.")
    print(f"\n  산출물: {outdir}")
    print("=" * 62)
    if not done:
        sys.exit(1)


if __name__ == "__main__":
    main()
