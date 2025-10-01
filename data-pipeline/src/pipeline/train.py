# src/pipeline/tasks/train.py
# -*- coding: utf-8 -*-
"""
Training task using IntegratedCovidTrainer (robust future-aware FE + dynamic forecast).
Loads engineered features from MLflow (or rebuilds quickly if needed).
"""
import os, sys, argparse
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _try_load_features(tracking_uri: str | None):
    # Try MLflowCovidPipeline first
    try:
        from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
        pipeline = MLflowCovidPipeline(tracking_uri=tracking_uri)
        if hasattr(pipeline, "load_latest_features"):
            return pipeline.load_latest_features()
    except Exception as e:
        print(f"[train] load_latest_features unavailable: {e}")
    # Generic util loader
    try:
        from src.utils.mlflow_utils import load_latest_artifact_dataframe
        return load_latest_artifact_dataframe(artifact_path="features/features.csv", tracking_uri=tracking_uri)
    except Exception as e:
        print(f"[train] generic features load failed: {e}")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="new_cases")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--tracking-uri", type=str, default=None)
    ap.add_argument("--rebuild-from-processed", action="store_true",
                    help="If set, rebuild features from latest processed instead of loading features artifact.")
    args = ap.parse_args()

    # Load features if available
    features_df, target_series = None, None
    if not args.rebuild_from_processed:
        try:
            features_df = _try_load_features(args.tracking_uri)
        except Exception as e:
            print(f"[train] feature load attempt failed: {e}")

    if features_df is None:
        # Rebuild quickly from processed data using your FE module
        try:
            from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
            pipeline = MLflowCovidPipeline(tracking_uri=args.tracking_uri)
            processed = pipeline.load_latest_processed()
        except Exception as e:
            print(f"[train] processed load failed via pipeline: {e}")
            processed = None
        if processed is None:
            from src.utils.mlflow_utils import load_latest_artifact_dataframe
            processed = load_latest_artifact_dataframe(artifact_path="data/processed.csv", tracking_uri=args.tracking_uri)
        from src.feature_engineering.feature_engineer import CovidFeatureEngineer
        fe = CovidFeatureEngineer(tracking_uri=args.tracking_uri)
        features_df, target_series = fe.engineer_features(processed, target_column=args.target, run_name="feature_engineering_for_train")

    # If engineer_features returned target separately, ensure column presence
    if target_series is not None and args.target not in features_df.columns:
        features_df = features_df.copy()
        features_df[args.target] = target_series.values

    # Train with existing features (no additional FE)
    from src.modeling.integrated_trainer import IntegratedCovidTrainer, TrainConfig

    cfg = TrainConfig(target_col=args.target, test_days=args.test_days, horizon=args.horizon)
    trainer = IntegratedCovidTrainer(cfg, tracking_uri=args.tracking_uri)
    res = trainer.train(features_df, target=args.target)

    print("[train] done.")
    print("Best model:", res["best_model"])
    print("Test metrics:", res["metrics"]["test"])

if __name__ == "__main__":
    main()