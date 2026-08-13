"""msa — 큐 프로토콜로 붙는 워커 껍데기.

``celery`` 를 임포트하므로 여기서 재노출하지 않는다. 단독 운영(웹 화면)에는
celery 가 필요 없고, `import face_anonymizer` 가 그걸 끌고 오면 안 된다.

    from face_anonymizer.msa.celery_app import app
"""
