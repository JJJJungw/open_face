# 024 — 워커에 없는 라이브러리를 오류 처리가 부르고 있었다

우리는 두 모습으로 돈다. 웹 서버(HTTP 를 받는다)와 큐 워커(화면 없이 남의 큐에서
일만 꺼내 온다). 워커 컨테이너에는 `fastapi` 를 **일부러 안 깐다** —
`requirements/worker.txt` 에 없다. HTTP 를 안 받으니 이미지에 넣을 이유가 없다.

그런데 저장소 코드가 S3 오류에 **HTTP 응답용 딱지**를 붙이려고 `service/errors`
를 임포트한다. 그 모듈은 최상위에서 `fastapi` 를 끈다.

```python
def wrap(e, what):
    from ..service import errors     # ← 여기서 fastapi 가 딸려 온다
```

## 무엇이 나쁜가

터지는 것 자체보다 **무엇으로 터지느냐**가 나쁘다.

```
실제로 일어난 일:  S3 접근 권한이 없습니다 (AccessDenied)
운영자가 보는 것:  No module named 'fastapi'
```

원인이 통째로 바뀐다. 권한 문제인데 받아 보는 사람은 설치가 잘못된 줄 알고
의존성을 뒤진다. 이 서비스가 계속 고쳐 온 것이 "조용한 실패" 인데, 이건 **시끄러운
거짓말**이라 더 나쁘다.

## 지금 설정으로는 안 터진다 — 그게 함정이다

확인했다. 워커는 버킷을 설정하지 않으므로 `get_store()` 가 `None` 이고 S3 를
아예 안 건드린다.

```
s3 모듈 임포트: OK (fastapi 없이도 열린다)
get_store() = None                  ← 여기서 끝. 안전하다
wrap() →  ImportError: No module named 'fastapi'
```

터지려면 워커에 버킷이 설정돼야 하는데, 그게 **`FA_WEIGHTS_SOURCE=s3`**
(가중치를 버킷에서 받기)다. `.env.example` 에 정상 옵션으로 적혀 있다. 그걸 켜는
순간 `weights.ensure()` 가 `get_store()` 를 살리고, 거기서 S3 오류가 하나라도
나면 이 길로 들어간다.

즉 **지금 없는 버그가 아니라 설정 하나 뒤에 있는 버그**다.

## 한 일

딱지는 **부가 정보**다. HTTP 상태 코드를 고르는 데 쓰는 것이고 워커에는 응답
자체가 없다. `S3Error.problem` 도 원래 `None` 이 기본값이다. 그러니 못 가져오면
안 붙이고 오류 메시지는 그대로 내보낸다.

```python
def _problems():
    try:
        from ..service import errors
    except ImportError:
        return None          # 워커 — 딱지 없이 간다
    return errors
```

`base.NotImplementedStore._no()` 도 같은 길을 탔으므로 같이 고쳤다.

고친 뒤 fastapi 가 없는 프로세스에서:

```
메시지 : 내려받지 못했습니다: Access Denied      ← 진짜 원인이 나온다
딱지   : None
```

서버에서는 그대로 붙는다(`s3_access_denied` / `s3_object_not_found`). 권한 문제와
키 오타와 네트워크 장애는 상태 코드가 달라야 한다.

## 왜 큰 수술을 안 했나

"문제 코드 상수를 fastapi 를 안 끄는 곳으로 내린다" 가 정석이고 처음엔 그렇게
보고 **범위가 커서 못 한다**고 적었다. 다시 보니 틀렸다. `wrap()` 이 쓰는 것은
Problem 객체 넷뿐이고, 그마저도 **없어도 되는 값**이다. 계층을 재배치하는 대신
"없으면 없는 대로" 를 명시하는 것으로 같은 결과를 얻는다.

역방향 임포트(`storage` → `service`) 자체는 남아 있다. 그건 구조 부채이지 버그가
아니고, 납품을 앞두고 계층을 옮기는 것이 새 버그를 만들 위험이 더 크다.

## 테스트를 새 프로세스에서 도는 이유

처음에는 같은 프로세스에서 `sys.modules` 를 지우고 fastapi 임포트를 막았다.
**통과했다 — 그런데 아무것도 안 지키고 있었다.** 부모 패키지(`service`)가
`errors` 를 속성으로 들고 있어서 재임포트가 일어나지 않았기 때문이다.

워커 컨테이너는 애초에 fastapi 가 **없는** 프로세스다. 그러니 새 프로세스에서
`meta_path` 로 막고 확인한다. 되돌리면(try/except 를 빼면) 실패한다.
