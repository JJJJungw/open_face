"""운영 지표.

작업 큐 대시보드가 무엇을 보여 줘야 하는지는 이미 답이 나와 있는 편이다.
Sidekiq 의 Web UI, Celery 의 Flower, RQ Dashboard, AWS Batch 콘솔이 공통으로
띄우는 것만 추리면 다섯 가지다.

1. **대기 깊이(depth)** — 몇 건이 기다리는가
2. **대기 지연(latency)** — 가장 오래 기다린 건이 얼마나 기다렸는가.
   깊이만 보면 100건이 1분 만에 빠지는 것과 3건이 두 시간째 멈춰 있는 것을
   구분할 수 없다. Sidekiq 이 이 값을 큐 화면의 머리로 두는 이유다.
3. **처리량(throughput)** — 최근 한 시간에 몇 건을 끝냈는가
4. **실패율과 재시도** — 실패가 코드 문제인지 인프라 문제인지 가르는 값
5. **워커/자원 상태** — 우리는 워커가 하나뿐이라 GPU 와 디스크가 그 자리다

여기에 이 서비스에만 있는 것을 하나 더한다. **폴더별 진척률** 이다. 위의
다섯은 전부 "지금 큐에 있는 것"만 말해 주는데, 데이터셋 비식별화에서 정작
궁금한 건 "전체 중 얼마나 남았나" 이고 그건 큐가 아니라 버킷에 있다.
"""

import os
import shutil
import subprocess
import time

from ..storage import naming

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
NVIDIA_TIMEOUT = 4


def is_video(key):
    return (os.path.splitext(key)[1].lower() in VIDEO_EXT
            and not naming.is_output(key))


def queue_metrics(jobs, now=None):
    """작업 목록에서 큐 지표를 뽑는다. jobs 는 Job 객체들.

    처리량은 최근 한 시간에 끝난 건수로 센다. 별도 시계열을 쌓지 않는 이유는
    작업 기록이 이미 디스크에 있고, 이 규모(수백 건)에서는 그걸 세는 게 가장
    싸고 정확하기 때문이다.
    """
    now = now or time.time()
    by = {k: [j for j in jobs if j.status == k]
          for k in ("queued", "running", "done", "failed", "cancelled")}

    waits = [now - j.created for j in by["queued"]]
    finished_recent = [j for j in by["done"] if j.finished > now - 3600]
    failed_recent = [j for j in by["failed"] if j.finished > now - 3600]
    durations = [j.result.get("seconds") for j in by["done"]
                 if isinstance(j.result, dict) and j.result.get("seconds")]
    retried = [j for j in jobs if j.attempts > 1]

    return {
        "depth": len(by["queued"]),
        "running": len(by["running"]),
        # 깊이만으로는 "빨리 빠지는 100건"과 "멈춰 있는 3건"을 구분 못 한다.
        "latency": round(max(waits)) if waits else 0,
        "done": len(by["done"]),
        "failed": len(by["failed"]),
        "cancelled": len(by["cancelled"]),
        "throughput_1h": len(finished_recent),
        "failed_1h": len(failed_recent),
        "retried": len(retried),
        "avg_seconds": round(sum(durations) / len(durations), 1) if durations else 0,
        "slowest_seconds": round(max(durations), 1) if durations else 0,
    }


def gpu_status(timeout=NVIDIA_TIMEOUT):
    """nvidia-smi 한 줄. GPU 가 없거나 실패하면 None.

    워커가 하나뿐이라 '워커 상태' 가 곧 이 GPU 의 상태다.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
             "memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0 or not p.stdout.strip():
        return None
    parts = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        return {"name": parts[0], "util": int(parts[1]),
                "mem_used": int(parts[2]), "mem_total": int(parts[3]),
                "temp": int(parts[4])}
    except ValueError:
        return None


def folder_progress(store, root=""):
    """입력 폴더별 전체/완료/남음.

    ``root`` 한 단계만 본다. 그 밑에 폴더가 있으면 폴더마다 한 줄, 없으면
    ``root`` 자체가 한 줄이다. 지금 버킷이 v1/input/{kbs,mbc,sbs}/ 처럼
    한 겹이라 이걸로 충분하고, 더 파고들면 왕복만 늘어난다.

    결과물이 있는지는 결과 프리픽스를 **한 번** 나열해서 대조한다. 객체마다
    HEAD 를 날리면 폴더 하나에 수백 번 왕복한다.
    """
    done_keys = store.processed_keys()
    folders, objects = store.list(root)
    rows = []

    def row(prefix, keys):
        fin = sum(1 for k in keys if store.output_key(k) in done_keys)
        return {"prefix": prefix,
                "name": prefix.rstrip("/").split("/")[-1] or "(최상위)",
                "total": len(keys), "done": fin, "remain": len(keys) - fin,
                "percent": round(100 * fin / len(keys)) if keys else 0}

    for f in folders:
        keys = [o["key"] for o in store.list(f)[1] if is_video(o["key"])]
        if keys:
            rows.append(row(f, keys))
    here = [o["key"] for o in objects if is_video(o["key"])]
    if here:
        rows.append(row(root, here))
    return sorted(rows, key=lambda x: -x["total"])
