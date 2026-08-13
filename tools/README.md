# tools — 손으로 돌리는 도구

테스트(`tests/`)와 다르다. 여기 있는 것은 **사람이 판단하려고 한 번씩 돌리는**
스크립트다. CI 가 돌리지 않고, 실패해도 빌드가 깨지지 않는다.

| 파일 | 하는 일 |
|---|---|
| `msa_smoke.py` | MSA 큐 왕복 한 바퀴 — 저쪽 없이 우리끼리 돌려 본다 |

## msa_smoke.py

붙일 곳(RebornStudio)이 아직 없으니 **저쪽이 할 일을 이 스크립트가 흉내 낸다** —
서명된 URL 을 대신하는 HTTP 서버, 잡을 큐에 넣는 발신자, 하트비트·완료를 받아
펜싱 토큰을 대조하는 수신자. 가운데에서 도는 것은 **진짜 우리 워커**다.

```bash
redis-server --daemonize yes                        # 브로커가 없으면
python tools/msa_smoke.py --input sample.mp4
python tools/msa_smoke.py --input sample.mp4 --repeat 5    # 평균을 보려면
```

가짜로 바꾸는 것이 하나도 없으므로 **여기서 나오는 처리 시간이 그대로 운영
숫자**다. 저쪽 KEDA 의 "대기 몇 건당 워커 한 대" 를 이 숫자로 정한다.

가중치도 GPU 도 없는 곳(노트북)에서 **배선만** 보려면:

```bash
python tools/msa_smoke.py --input sample.mp4 --fake-detector
```

이때도 큐·전송·하트비트·펜싱은 전부 진짜 경로를 탄다. 처리 시간만 의미가 없다.

로그는 `--outdir/logs/` 에 남는다. 실패했을 때 있는 것이 그것뿐이라 지우지 않는다.
