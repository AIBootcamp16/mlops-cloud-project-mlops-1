
# src/pipeline/orchestrate.py
# -*- coding: utf-8 -*-
"""
Orchestrate the 4 tasks sequentially for local/manual runs.
Airflow DAG에서는 각 task 모듈을 개별 BashOperator로 호출하세요.
"""
import os, sys, argparse, subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_cmd(cmd: list[str], env=None):
    print("+", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", type=str, default=None)
    ap.add_argument("--target", type=str, default="new_cases")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.tracking_uri: env["MLFLOW_TRACKING_URI"] = args.tracking_uri

    if not args.skip_collect:
        run_cmd([sys.executable, "-m", "src.pipeline.collect", "--run-name", "collection"], env)

    run_cmd([sys.executable, "-m", "src.pipeline.preprocess", "--from-latest", "--run-name", "preprocessing"], env)
    run_cmd([sys.executable, "-m", "src.pipeline.fe", "--run-name", "feature_engineering_v1", "--target", args.target], env)
    run_cmd([sys.executable, "-m", "src.pipeline.train", "--target", args.target, "--test-days", str(args.test_days), "--horizon", str(args.horizon)], env)

if __name__ == "__main__":
    main()
