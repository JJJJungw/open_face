# 019 — CLI 가 죽어 있었는데 테스트 412개는 통과하고 있었다

납품 비트레이트를 실제 방송 클립으로 재 보려고 EC2 에서 `face-anonymize` 를
쳤다. 그게 첫 줄에서 죽었다.

```
$ .venv/bin/face-anonymize /tmp/ttttt.mp4 -o /tmp/out.mp4
Traceback (most recent call last):
  File ".../cli.py", line 153, in main
    from .pipeline import VideoAnonymizer
ModuleNotFoundError: No module named 'face_anonymizer.pipeline'
```

**README 의 첫 사용 예제다.** 남이 클론해서 제일 먼저 치는 명령이기도 하다.

## 원인

`core/` 로 폴더를 나눌 때 `cli.py` 의 임포트 하나가 옛 경로에 남았다. 같은 파일
**24번 줄은 제대로 고쳐져 있다.**

```python
from .core.pipeline import (DetectionSanityError, VideoOpenError,   # 24행 — 고쳐짐
                            VideoWriteError)
...
        from .pipeline import VideoAnonymizer                        # 153행 — 안 고쳐짐
```

차이는 하나다. **153행은 함수 안에 있는 지연 임포트다.** `--help` 나 인자 오류에서
torch(2GB)를 끌고 오지 않으려고 일부러 그렇게 둔 것인데, 그 대가로 **모듈을
임포트해 보는 것으로는 안 걸린다.** 파이썬은 그 줄에 도달할 때까지 아무 말도 안
한다.

## 왜 아무도 몰랐나

테스트가 412개 있었고 전부 통과했다. 그런데 **`cli.main()` 을 부르는 테스트가 하나도
없었다.** `tests/` 에 `test_cli.py` 자체가 없었다.

파이프라인은 `VideoAnonymizer` 를 직접 만들어 검증하고, 서버는 `TestClient` 로
라우트를 두드린다. CLI 만 아무도 안 지나갔다. 그래서 **가장 많이 쓰이는 진입점이
가장 안 지켜지고 있었다.**

여기에 013 과 같은 냄새가 있다. 그때는 화면 세 곳이 모델에 대해 다른 말을 했고,
그 뒤에 아무도 안 밟은 죽은 경로가 있었다. 이번에도 같다 — **안 밟히는 경로는
썩는다.**

## 한 일

임포트를 고쳤다. 한 글자다. 중요한 건 그 뒤다.

**`tests/test_cli.py` 를 만들었다.** 가짜 파이프라인을 꽂아 `main()` 을 실제로
끝까지 돌린다. 지연 임포트는 부르는 순간 풀리므로, 원본 모듈의
`VideoAnonymizer` 를 갈아 끼우면 그대로 잡힌다.

- `test_the_readme_first_example_actually_runs` — `face-anonymize input.mp4`
- `test_cli_options_reach_the_pipeline` — 인자를 받아 놓고 안 넘기면 조용히
  기본값으로 처리된다. `--conf 0.15` 를 줬는데 0.25 로 도는 것은 알아채기 어렵다.
- `test_bad_arguments_do_not_crash` — 잘못된 인자는 트레이스백이 아니라 종료 코드

**그리고 이 종류를 전수로 막았다.** `test_no_stale_relative_imports_anywhere` 가
패키지 안의 모든 상대 임포트를 AST 로 훑어 그 모듈이 실제로 존재하는지 본다.
고친 것은 한 건이지만 **함수 안에 숨은 상대 임포트가 스물아홉 개 더 있다.**
폴더를 또 옮기면 그중 아무거나 같은 방식으로 죽고, 그때도 테스트는 통과한다.

되돌려서 확인했다. 임포트를 원래대로 되돌리면 네 개가 전부 실패한다.

## 배운 것

**"임포트가 되는가" 와 "그 경로가 도는가" 는 다른 질문이다.** 지연 임포트는 후자로만
검증된다. 성능을 위해 임포트를 함수 안으로 내리는 것은 정당하지만, 그 순간 그
줄은 **테스트가 실제로 그 함수를 부르지 않으면 아무도 안 보는 코드**가 된다.

이번 것은 018 을 확인하다가 우연히 걸렸다. 실제 장비에서 실제 파일로 한 번
돌려 보지 않았으면 계속 몰랐을 것이다. 018 에도 같은 교훈이 적혀 있다 — 설정값이
명령줄에 실리는 것까지만 보고 그 명령줄이 무엇을 만드는지는 안 봤던 것.
**끝까지 한 번 돌려 보는 것을 대신할 수 있는 것은 없다.**
