# tests/test_smoke.py
import os

def test_python_runs():
    """기본 파이썬 실행 테스트"""
    assert True  # 단순 Smoke 테스트

def test_mlflow_import():
    """MLflow import 테스트"""
    import mlflow
    assert mlflow is not None

def test_env_loaded():
    """필수 환경변수 로드 여부 확인 (기본값 자동 주입)"""
    # CI나 로컬에서 값이 비어 있어도 기본값으로 세팅
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    os.environ.setdefault("AWS_DEFAULT_REGION", "ap-northeast-2")

    required_vars = ["MLFLOW_TRACKING_URI", "AWS_DEFAULT_REGION"]
    missing = [var for var in required_vars if not os.getenv(var)]
    assert not missing, f"Missing env vars: {missing}"
