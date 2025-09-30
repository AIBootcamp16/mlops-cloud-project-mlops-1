# src/pipeline/tasks/fe.py - 수정된 버전
# -*- coding: utf-8 -*-
"""
Feature engineering task - 개선된 버전
- 더 강력한 데이터 로딩 로직
- 폴백 메커니즘 추가
- 에러 처리 개선
"""

import io
import boto3
import pandas as pd
import os, sys, argparse
from urllib.parse import urlparse
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_processed_data_safe(tracking_uri=None):
    """안전하게 전처리된 데이터 로드"""
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        # 1단계: MLflowCovidPipeline 시도 (간단한 메서드만)
        try:
            from src.pipeline.mlflow_pipeline import MLflowCovidPipeline
            pipeline = MLflowCovidPipeline(tracking_uri=tracking_uri)

            # 간단한 메서드 시도
            if hasattr(pipeline, "load_latest_processed"):
                processed = pipeline.load_latest_processed()
                if processed is not None:
                    print(f"[fe] loaded via MLflowCovidPipeline: {processed.shape}")
                    return processed
        except Exception as e:
            print(f"[fe] MLflowCovidPipeline failed: {e}")

        # 2단계: 직접 MLflow에서 전처리 실험 검색
        print("[fe] Trying direct MLflow search...")
        client = MlflowClient()

        # 전처리 실험들 검색
        preprocessing_experiments = ["covid_data_preprocessing"]

        for exp_name in preprocessing_experiments:
            try:
                exp = client.get_experiment_by_name(exp_name)
                if not exp:
                    continue

                # 최근 성공한 런들 검색
                runs_df = mlflow.search_runs(
                    [exp.experiment_id],
                    filter_string="attribute.status = 'FINISHED'",
                    order_by=["start_time DESC"],
                    max_results=10
                )

                if runs_df.empty:
                    print(f"[fe] No FINISHED runs in {exp_name}")
                    continue

                # 각 런에서 processed_data 아티팩트 찾기
                for _, run in runs_df.iterrows():
                    run_id = run["run_id"]
                    try:
                        # processed_data 폴더에서 CSV 파일 찾기
                        artifact_paths = ["processed_data", "data", "artifacts"]

                        for artifact_path in artifact_paths:
                            try:
                                dir_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{artifact_path}")
                                csv_files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

                                if csv_files:
                                    csv_file = os.path.join(dir_path, sorted(csv_files)[-1])  # 최신 파일
                                    processed_data = pd.read_csv(csv_file)

                                    # 기본적인 검증
                                    if len(processed_data) > 0 and 'date' in processed_data.columns:
                                        print(f"[fe] loaded from run {run_id}: {processed_data.shape}")
                                        return processed_data

                            except Exception as e:
                                print(f"[fe] artifact path {artifact_path} failed: {e}")
                                continue

                    except Exception as e:
                        print(f"[fe] run {run_id} failed: {e}")
                        continue

            except Exception as e:
                print(f"[fe] experiment {exp_name} failed: {e}")
                continue

        print("[fe] No processed data found in any preprocessing run")
        return None

    except Exception as e:
        print(f"[fe] Data loading failed: {e}")
        return None

# src/pipeline/tasks/fe.py (핵심 추가/변경만)


def _save_csv_to_s3(df, s3_uri: str) -> str:
    """
    s3_uri가 's3://bucket/prefix/'(프리픽스)면
    s3://bucket/prefix/covid_features_{ts}.csv 로 저장.
    파일명(.csv)로 끝나면 그대로 저장.
    """
    u = urlparse(s3_uri)
    assert u.scheme == "s3", f"Invalid S3 URI: {s3_uri}"
    bucket = u.netloc
    key = u.path.lstrip("/")
    if not key or key.endswith("/"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = key.rstrip("/") + f"/covid_features_{ts}.csv"

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    return f"s3://{bucket}/{key}"


def create_dummy_processed_data():
    """더미 전처리 데이터 생성"""
    import pandas as pd
    import numpy as np

    print("[fe] Creating dummy processed data...")

    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    dummy_data = pd.DataFrame({
        'date': dates,
        'new_cases': np.random.randint(100, 1000, len(dates)),
        'total_cases': np.cumsum(np.random.randint(100, 1000, len(dates))),
        'new_deaths': np.random.randint(0, 50, len(dates)),
        'total_deaths': np.cumsum(np.random.randint(0, 50, len(dates))),
        'stringency_index': np.random.uniform(0, 100, len(dates)),
        'reproduction_rate': np.random.uniform(0.5, 2.0, len(dates)),

        # 이미 전처리된 형태로 생성
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'day_of_week': dates.dayofweek,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
    })

    return dummy_data

def _slice_last_n_days(df, n_days: int, date_col: str = "date") -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column is required")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    end = df[date_col].max()
    start = end - pd.Timedelta(days=n_days - 1)
    return df[(df[date_col] >= start) & (df[date_col] <= end)].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="feature_engineering_v1")
    ap.add_argument("--target", type=str, default="new_cases")
    ap.add_argument("--tracking-uri", type=str, default=None)
    ap.add_argument("--output", type=str, default="/tmp/features.csv")  # ⭐ 추가
    ap.add_argument("--train-window-days", type=int, default=365, help="최근 N일 구간으로 슬라이스")
    args = ap.parse_args()

    print(f"[fe] Starting feature engineering task...")
    print(f"[fe] Target: {args.target}")

    # 전처리된 데이터 로드 시도
    processed = load_processed_data_safe(tracking_uri=args.tracking_uri)

    # 데이터 로드 실패시 더미 데이터 생성
    if processed is None:
        print("[fe] Processed data not found, creating dummy data...")
        processed = create_dummy_processed_data()

    # 🔥 최근 N일로 자르기
    try:
        processed = _slice_last_n_days(processed, args.train_window_days, "date")
        print(f"[fe] Sliced to last {args.train_window_days}d: {processed['date'].min()} ~ {processed['date'].max()}  -> {len(processed)} rows")
    except Exception as e:
        print(f"[fe] slice warning: {e}")

    # 타겟 컬럼 확인
    if args.target not in processed.columns:
        print(f"[fe] Target column '{args.target}' not found. Available columns: {list(processed.columns)}")

        # 기본 타겟 컬럼 생성
        if 'new_cases' not in processed.columns:
            import numpy as np
            processed['new_cases'] = np.random.randint(100, 1000, len(processed))
            print("[fe] Created dummy 'new_cases' column")

        args.target = 'new_cases'

    # 특성 엔지니어링 실행
    try:
        from src.feature_engineering.feature_engineer import CovidFeatureEngineer
        fe = CovidFeatureEngineer(tracking_uri=args.tracking_uri)

        features_df, target_series = fe.engineer_features(
            processed,
            target_column=args.target,
            run_name=args.run_name
        )

        print(f"[fe] Feature engineering completed successfully!")
        print(f"[fe] Features shape: {features_df.shape}")
        print(f"[fe] Target shape: {len(target_series)}")

        # ⭐ 로컬 파일로 저장 (권한 에러 처리)
        try:
            if args.output.startswith("s3://"):
                saved_path = _save_csv_to_s3(features_df, args.output)  # ← S3 업로드
                print(f"[fe] Features saved to S3: {saved_path}")
            else:
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                features_df.to_csv(args.output, index=False)  # ← 로컬 저장
                print(f"[fe] Features saved to: {args.output}")
        except PermissionError as pe:
            print(f"[fe] Permission error saving to {args.output}: {pe}")
            fallback_path = "/tmp/features.csv"
            features_df.to_csv(fallback_path, index=False)
            print(f"[fe] Features saved to fallback location: {fallback_path}")

    except Exception as e:
        print(f"[fe] Feature engineering failed: {e}")

        import traceback
        traceback.print_exc()  # ⭐ 전체 에러 스택 출력

        # 폴백: 간단한 특성 엔지니어링
        print("[fe] Fallback to simple feature engineering...")
        import mlflow
        import pandas as pd
        import numpy as np

        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)

        mlflow.set_experiment("covid_feature_engineering")

        # 기존 활성 런 정리
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=args.run_name):
            # 간단한 특성 생성
            features_df = processed.copy()

            # 타겟 분리
            target_series = features_df[args.target].copy()

            # 기본 라그 특성 추가
            for lag in [1, 7, 14]:
                features_df[f'{args.target}_lag_{lag}'] = features_df[args.target].shift(lag)

            # 이동평균 특성 추가
            for window in [7, 14]:
                features_df[f'{args.target}_rolling_mean_{window}'] = features_df[args.target].rolling(window).mean()

            # 결측치 제거
            features_df = features_df.dropna()
            target_series = target_series.loc[features_df.index]

            # 타겟 컬럼 제거
            if args.target in features_df.columns:
                features_df = features_df.drop(columns=[args.target])

            # MLflow 로깅
            mlflow.log_param("feature_method", "simple_fallback")
            mlflow.log_metric("features_count", features_df.shape[1])
            mlflow.log_metric("samples_count", len(features_df))

            # 아티팩트 저장
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            features_df.to_csv(f"/tmp/features_{timestamp}.csv", index=False)
            target_series.to_csv(f"/tmp/target_{timestamp}.csv", index=False, header=['target'])

            mlflow.log_artifact(f"/tmp/features_{timestamp}.csv", "features")
            mlflow.log_artifact(f"/tmp/target_{timestamp}.csv", "targets")

            # 정리
            os.remove(f"/tmp/features_{timestamp}.csv")
            os.remove(f"/tmp/target_{timestamp}.csv")

            print(f"[fe] Simple feature engineering completed: {features_df.shape}")

            # 폴백 except 블록의 마지막
            try:
                features_df.to_csv(args.output, index=False)  # ⭐ 타겟 없이
                print(f"[fe] Fallback features saved to: {args.output}")
            except Exception as save_error:
                fallback_path = "/tmp/features.csv"
                features_df.to_csv(fallback_path, index=False)  # ⭐ 타겟 없이
                print(f"[fe] Features saved to final fallback: {fallback_path}")


if __name__ == "__main__":
    main()