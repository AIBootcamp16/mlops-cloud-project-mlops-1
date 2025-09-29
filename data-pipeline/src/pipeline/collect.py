
# src/pipeline/tasks/collect.py
# -*- coding: utf-8 -*-
"""
Data collection task.
Priority: use MLflowCovidPipeline if available; fallback to covid_collector module.
Logs raw data artifact into MLflow inside the pipeline (per your existing code).
"""
import os, sys, argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None, help="MLflow run name for collection")
    ap.add_argument("--tracking-uri", type=str, default=None)
    args = ap.parse_args()

    # Prefer MLflowCovidPipeline
    try:
        from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
        pipeline = MLflowCovidPipeline(tracking_uri=args.tracking_uri)
        if hasattr(pipeline, "run_collection"):
            result = pipeline.run_collection(run_name=args.run_name or "collection")
        elif hasattr(pipeline, "run_full_pipeline"):
            # As a minimal collection if no separate function; full pipeline will also preprocess
            result = pipeline.run_full_pipeline(collection_run_name=args.run_name or "collection")
        else:
            raise AttributeError("MLflowCovidPipeline has no run_collection or run_full_pipeline")
        print("[collect] done.")
        return
    except Exception as e:
        print(f"[collect] MLflowCovidPipeline path failed: {e}. Fallback to covid_collector.")

    # Fallback collector
    try:
        from src.collectors.covid_collector import CovidDataCollector
        collector = CovidDataCollector(tracking_uri=args.tracking_uri)
        collector.run(run_name=args.run_name or "collection")
        print("[collect] done via CovidDataCollector.")
    except Exception as e2:
        raise RuntimeError(f"Collection failed in both methods: {e2}")

if __name__ == "__main__":
    main()
