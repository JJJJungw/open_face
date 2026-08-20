# 026 — 검사 둘이 엉뚱한 이유로 통과하고 있었다

내 기계에서 458건 전부 통과한 것을 EC2 에서 돌리자 둘이 깨졌다. **둘 다 그날
바꾼 코드와 아무 상관이 없었다.** 원래부터 그 자리에 있던 것이고, 기계가 바뀌자
드러났을 뿐이다.

```
FAILED tests/test_storage_contract.py::test_an_s3_error_survives_without_fastapi
FAILED tests/test_storage_setup.py::test_the_setting_survives_a_restart_but_the_keys_do_not
```

---

## 하나 — 차단이 안 먹는데 통과하고 있었다

024 에서 만든 검사다. 워커 컨테이너에는 `fastapi` 가 없으므로, 그 상황에서
`wrap()` 이 딱지를 못 가져와도 진짜 원인이 살아남는지를 본다. 없는 상황을
만들려고 임포트를 막았다.

```python
class Block:
    def find_module(self, name, path=None): ...
    def load_module(self, name): raise ImportError(...)
```

**`find_module` / `load_module` 은 파이썬 3.12 에서 제거됐다.** 3.4 부터 폐기
예고돼 있었고 3.12 에서 실제로 빠졌다. 그래서 3.12 인 EC2 에서는 이 훅이 아무것도
안 막는다 — `fastapi` 가 그대로 임포트되고, 딱지가 붙고, `problem is None` 단언이
깨졌다.

실제로 재 봤다.

```
python3.11   Old: 차단 성공    New: 차단 성공
python3.12   Old: 차단 실패    New: 차단 성공
python3.13   Old: 차단 실패    New: 차단 성공
```

**여기서 운이 좋았다.** 이 검사의 단언이 "딱지가 **없어야** 한다" 라서, 차단이
안 먹자 실패로 드러났다. 만약 단언이 반대 방향이었으면 3.12 에서는 아무것도 안
보면서 **초록불로 통과**했을 것이다. 검사가 죽는 것보다 살아 있는 척하는 게
훨씬 나쁘다.

### 해결

`find_spec` 으로 바꾸고, **차단이 먹었는지를 먼저 확인한다.**

```python
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return None

try:
    import fastapi
except ImportError:
    pass
else:
    raise AssertionError("차단이 안 먹었다 — 이 검사는 아무것도 안 본다")
```

앞으로 파이썬이 또 훅을 갈아 치우면, 조용히 통과하는 대신 여기서 큰 소리로
멈춘다.

---

## 둘 — "첫 실행" 검사가 그 기계의 `.env` 를 물려받았다

`fresh` 픽스처는 "아직 아무것도 안 정해진 서버" 를 만든다. 그런데 환경 변수를
안 비웠다.

```python
again = providers.StorageConfig.from_env()
assert again.bucket == GOOD          # 'good-bucket'
```

```
AssertionError: 'ax-mbc-label-data-storage' == 'good-bucket'
```

`from_env()` 는 **환경 변수를 저장 파일보다 먼저 본다.** 그건 맞는 순서다 —
`.env` 를 고쳤는데 화면에서 눌러 둔 옛 값이 이기면, 사람은 있지도 않은 문제를
찾게 된다(그 자체가 예전에 고친 것이다). 문제는 검사 쪽이었다. 진짜 서버에는
`FA_S3_BUCKET` 이 실제 버킷으로 박혀 있으므로, 그 기계에서 돌리는 순간 "첫
실행" 이 첫 실행이 아니게 된다.

**검사가 코드가 아니라 그 기계의 설정을 보고 있었다.** 개발 기계에는 그 변수가
없어서 계속 통과했다.

### 해결

픽스처가 `from_env()` 가 읽는 이름을 전부 비운다.

```python
for k in ("FA_S3_BUCKET", "FA_S3_REGION", "FA_S3_ENDPOINT",
          "FA_S3_ROOT_PREFIX", "FA_S3_OUTPUT_PREFIX",
          "FA_STORAGE_PROVIDER", "FA_STORAGE_STORE"):
    monkeypatch.delenv(k, raising=False)
```

## 검증

EC2 를 흉내 내서 개발 기계에서 재현했다.

```
FA_S3_BUCKET=ax-mbc-label-data-storage FA_STORAGE_PROVIDER=ncp \
  python -m pytest tests/test_storage_setup.py -q
```

고치기 전에는 같은 자리에서 깨지고, 고친 뒤에는 통과한다.

## 배운 것

**검사는 자기가 아무것도 안 보고 있다는 걸 스스로 말하지 않는다.** 환경을
조작해서 상황을 만드는 검사는, 그 조작이 실제로 먹었는지를 같이 확인해야 한다.
안 그러면 조작 방법이 낡는 순간 초록불만 남는다.

**개발 기계는 깨끗해서 위험하다.** 실제 서버에만 있는 환경 변수는 검사를 통과
시키는 게 아니라 검사의 의미를 바꾼다. "아무것도 안 정해진 상태" 를 만드는
픽스처는 세팅뿐 아니라 **지우기**까지 해야 한다.

**한 기계에서만 돌린 초록불은 초록불이 아니다.** 파이썬 3.11 과 3.12 의 차이
하나로 검사 하나가 조용히 죽어 있었다.
