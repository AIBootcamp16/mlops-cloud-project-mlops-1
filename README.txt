1. data-pipeline/src/config/setting.py 에서 설정 (TRACKING_URI : mlruns 생성 경로)

2. data-pipeline/src/main.py 데이터 수집 -> 모델 학습 -> 모델 등록 파이프라인

3. data-pipeline/src/server/server.py 서버 파일 터미널에서 data-pipeline/src/server 경로에서 uvicorn server:app --host 0.0.0.0 --port 5432

4. api 요청 curl "http://localhost:5432/date?date=2025-12-30"
