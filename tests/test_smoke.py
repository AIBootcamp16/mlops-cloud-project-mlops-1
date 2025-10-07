# tests/test_smoke.py
import os
import pytest

def test_python_runs():
    """기본 파이썬 실행 테스트"""
    assert True  # 단순 Smoke 테스트

def test_mlflow_import():
    """MLflow import 테스트"""
    import mlflow
    assert mlflow is not None

def test_env_loaded():
    """필수 환경변수 로드 여부 확인"""
    required_vars = ["MLFLOW_TRACKING_URI", "AWS_DEFAULT_REGION"]
    missing = [var for var in required_vars if os.getenv(var) is None]
    assert not missing, f"Missing env vars: {missing}"
