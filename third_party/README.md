# third_party — 외부 리포

`scripts/setup_weights.py` 가 [clibdev/YOLO-FaceV2](https://github.com/clibdev/YOLO-FaceV2)
를 여기에 클론한다.

## 왜 필요한가

가중치 파일(`.pt`)이 파이썬 pickle 이라, 불러올 때 그 리포의 `models/` ·
`utils/` 모듈이 임포트되어야 한다. 없으면 `ModuleNotFoundError` 로 검출기
생성이 실패한다. 우리 코드가 직접 쓰는 건 아니고, 체크포인트를 푸는 데 필요하다.

`requirements/base.txt` 의 pandas · matplotlib · seaborn · thop 같은 것들도 같은
이유로 들어가 있다. 그 리포의 임포트 사슬(`models/common.py` →
`utils/general.py` → `utils/plots.py`)이 요구한다.

커밋하지 않는다(`.gitignore`). 이 폴더는 `.gitkeep` 으로 경로만 유지한다.
