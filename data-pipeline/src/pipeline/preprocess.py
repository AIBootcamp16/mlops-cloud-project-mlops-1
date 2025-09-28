
# src/pipeline/tasks/preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocessing task.
Loads the latest collection output from MLflow and produces processed dataset,
either via MLflowCovidPipeline or direct data_preprocessor module.
"""
import os, sys, argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="preprocessing")
    ap.add_argument("--from-latest", action="store_true", help="Use latest collection run from MLflow")
    ap.add_argument("--tracking-uri", type=str, default=None)
    args = ap.parse_args()

    # Prefer MLflowCovidPipeline convenience
    try:
        from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
        pipeline = MLflowCovidPipeline(tracking_uri=args.tracking_uri)
        if args.from_latest and hasattr(pipeline, "run_preprocessing_from_latest_collection"):
            # 메서드가 어떤 키워드를 받는지 감지해서 호출
            import inspect
            sig = inspect.signature(pipeline.run_preprocessing_from_latest_collection)
            if "run_name" in sig.parameters:
                processed = pipeline.run_preprocessing_from_latest_collection(run_name = args.run_name, from_latest = args.from_latest)
            elif "preprocessing_run_name" in sig.parameters:
                processed = pipeline.run_preprocessing_from_latest_collection(
                preprocessing_run_name = args.run_name, from_latest = args.from_latest)
            else:
                raise TypeError("run_preprocessing_from_latest_collection signature not recognized")
        elif hasattr(pipeline, "run_preprocessing"):
            import inspect
            sig = inspect.signature(pipeline.run_full_pipeline)
            if "run_name" in sig.parameters:
                processed = pipeline.run_full_pipeline(run_name=args.run_name)
            elif "preprocessing_run_name" in sig.parameters:
                processed = pipeline.run_full_pipeline(preprocessing_run_name=args.run_name)
            else:
                processed = pipeline.run_full_pipeline()
        elif hasattr(pipeline, "run_full_pipeline"):
            processed = pipeline.run_full_pipeline(run_name=args.run_name)
        else:
            raise AttributeError("Suitable preprocessing method not found in MLflowCovidPipeline")
        print("[preprocess] done.")
        return
    except Exception as e:
        print(f"[preprocess] MLflowCovidPipeline path failed: {e}. Fallback to data_preprocessor.")

    # Fallback direct module
    try:
        from src.data_processing.data_preprocessor import CovidDataPreprocessor
        pre = CovidDataPreprocessor(tracking_uri=args.tracking_uri)
        pre.run(run_name=args.run_name, from_latest=args.from_latest)
        print("[preprocess] done via CovidDataPreprocessor.")
    except Exception as e2:
        raise RuntimeError(f"Preprocessing failed in both methods: {e2}")

if __name__ == "__main__":
    main()
