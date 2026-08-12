# service — HTTP API · 웹 UI · 운영 지표

코어를 감싸 서비스로 만드는 층. 큐, 작업 상태, 오류 분류, 화면이 여기 있다.

**코어는 이 패키지를 모른다.** 의존은 `service` → `core` 한 방향뿐이다.

## 파일

| 파일 | 하는 일 |
|---|---|
| `server.py` | FastAPI 앱. 진입점은 `POST /api/jobs` 하나 |
| `errors.py` | RFC 9457 problem+json. 오류 31종의 코드·문구·재시도 여부 |
| `metrics.py` | 큐 지표(깊이·대기 지연·처리량)와 폴더별 진척률, GPU·디스크 상태 |
| `webui.py` | `static/index.html` 을 읽어 오는 로더 |
| `static/index.html` | 화면 전부. HTML·CSS·JS 한 파일, 빌드 도구도 CDN 도 없다 |

## 실행

```bash
uvicorn face_anonymizer.service.server:app --host 127.0.0.1 --port 8000
```

## 못 박아 둔 것

**한 번에 한 편.** GPU 한 장에 검출기 하나라 워커 스레드도 하나다. 프로세스를
여러 개 띄워도 파일 락으로 직렬화한다.

**대기열은 개수로 막지 않는다.** 폴더 하나에 수천 건이 정상이다. 진짜 제약은
디스크라서 여유 공간으로 막는다(507).

**한 건이 거절돼도 나머지는 받는다.** 수백 건 중 키 하나가 오타라고 전체를
되돌리면 호출하는 쪽이 무엇이 들어갔는지 알 수 없다.

**오류는 코드로 남긴다.** 사람이 읽는 문구와 별개로 `code` 가 안정된 식별자다.
`GET /api/problems` 로 전체 목록을 볼 수 있다.
