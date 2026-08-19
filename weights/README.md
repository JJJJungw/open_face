# weights — 모델 가중치

`scripts/setup_weights.py` 가 여기에 `yolo-facev2.pt` 를 받아 둔다.

```bash
python scripts/setup_weights.py
```

가중치 자체는 커밋하지 않는다(`.gitignore`). 이 폴더는 `.gitkeep` 으로 경로만
유지한다 — 없으면 다운로드가 실패한다.
