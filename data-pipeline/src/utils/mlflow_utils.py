"""MLflow 유틸리티 함수들"""
import mlflow
import mlflow.data
import pandas as pd
from typing import Dict, Any, Optional
import json
import os
import tempfile
from datetime import datetime
from mlflow.tracking import MlflowClient


class MLflowManager:
    """MLflow 실험 관리 클래스"""

    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        exp = client.get_experiment_by_name(self.experiment_name) or client.get_experiment(
            mlflow.get_experiment_by_name(self.experiment_name).experiment_id)
        print(f"[diag] manager.exp={self.experiment_name}")
        try:
            ex2 = mlflow.get_experiment_by_name(self.experiment_name)
            print(f"[diag] artifact_location={ex2.artifact_location if ex2 else 'N/A'}")
        except Exception as e:
            print("[diag] exp lookup failed:", e)

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """MLflow run 시작"""
        import mlflow, os
        print("[diag] mlflow.__version__ =", mlflow.__version__)
        print("[diag] tracking_uri =", mlflow.get_tracking_uri())
        print("[diag] active_run_artifact_uri =", mlflow.get_artifact_uri() if mlflow.active_run() else "(no run)")
        print("[diag] env.MLFLOW_TRACKING_URI =", os.getenv("MLFLOW_TRACKING_URI"))
        from mlflow.tracking import MlflowClient
        exp = MlflowClient().get_experiment_by_name("covid_data_collection")
        print("[diag] experiment.artifact_location =", (exp.artifact_location if exp else "N/A"))
        if run_name is None:
            run_name = f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M')}"

        run = mlflow.start_run(run_name=run_name)
        print(f"[MLflowManager] Started new run: {run_name} ({run.info.run_id})")
        return run

    def log_data_metrics(self, data: pd.DataFrame, prefix: str = "") -> None:
        """데이터 품질 메트릭 로깅"""
        metrics = {
            f"{prefix}total_rows": len(data),
            f"{prefix}total_columns": len(data.columns),
            f"{prefix}missing_cols_count": data.isnull().any().sum(),
            f"{prefix}complete_rows": len(data.dropna()),
            f"{prefix}missing_percentage": (data.isnull().any().sum() / len(data.columns)) * 100
        }

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    def log_column_availability(self, data: pd.DataFrame, key_columns: list) -> None:
        """주요 컬럼 데이터 가용성 로깅"""
        for col in key_columns:
            if col in data.columns:
                availability = data[col].notna().sum() / len(data) * 100
                mlflow.log_metric(f"{col}_availability_pct", availability)

    def log_dataset(self, data: pd.DataFrame, source_url: str, name: str, context: str = "raw_data") -> None:
        """데이터셋을 MLflow Dataset으로 로깅"""
        dataset = mlflow.data.from_pandas(
            data,
            source=source_url,
            name=name,
            digest="v1.0"
        )
        mlflow.log_input(dataset, context=context)

    def save_and_log_artifact(self, data: pd.DataFrame, filename: str, artifact_path: str) -> str:
        """데이터를 CSV로 저장하고 MLflow 아티팩트로 등록"""
        # 임시 디렉터리에 저장
        tmp_dir = tempfile.gettempdir()
        full_path = os.path.join(tmp_dir, filename)

        # CSV 파일 저장
        data.to_csv(full_path, index=False)

        # MLflow 아티팩트로 등록
        mlflow.log_artifact(full_path, artifact_path)

        # 임시 파일 정리
        os.remove(full_path)

        return f"{artifact_path}/{filename}"

    def log_metadata(self, metadata: Dict[str, Any], filename: str, artifact_path: str = "metadata") -> None:
        """메타데이터를 JSON으로 저장하고 아티팩트로 등록"""
        tmp_dir = tempfile.gettempdir()
        full_path = os.path.join(tmp_dir, filename)

        with open(full_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        mlflow.log_artifact(full_path, artifact_path)

        # 임시 파일 정리
        os.remove(full_path)


def create_data_metadata(data: pd.DataFrame, collection_info: Dict[str, Any]) -> Dict[str, Any]:
    """데이터 메타데이터 생성"""
    return {
        'collection_info': {
            'timestamp': datetime.now().isoformat(),
            'collector': 'MLflow_COVID_Collector',
            **collection_info
        },
        'data_summary': {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d') if 'date' in data.columns else None,
                'end': data['date'].max().strftime('%Y-%m-%d') if 'date' in data.columns else None
            },
            'missing_data_columns': int(data.isnull().any().sum()),
            'complete_rows': len(data.dropna()),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024 ** 2, 2)
        },
        'column_summary': {
            col: {
                'dtype': str(data[col].dtype),
                'missing_count': int(data[col].isnull().sum()),
                'availability_pct': round((data[col].notna().sum() / len(data)) * 100, 1)
            } for col in data.columns
            if col not in ['collected_at', 'data_source', 'collector']  # 메타 컬럼 제외
        }
    }


def load_latest_artifact_dataframe(artifact_path: str, tracking_uri: Optional[str] = None) -> Optional[pd.DataFrame]:
    """최신 아티팩트에서 DataFrame 로드"""
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()

        # 모든 실험에서 최신 성공한 런 찾기
        experiments = client.search_experiments()
        all_runs = []

        for exp in experiments:
            runs_df = mlflow.search_runs(
                [exp.experiment_id],
                filter_string="attribute.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=10
            )
            if not runs_df.empty:
                all_runs.append(runs_df)

        if not all_runs:
            return None

        # 모든 런을 합치고 최신순 정렬
        combined_runs = pd.concat(all_runs, ignore_index=True)
        combined_runs = combined_runs.sort_values('start_time', ascending=False)

        # 아티팩트 경로에서 CSV 파일 찾기
        for _, run in combined_runs.iterrows():
            run_id = run["run_id"]
            try:
                # 다양한 아티팩트 경로 시도
                possible_paths = [
                    artifact_path,
                    artifact_path.replace("data/", ""),
                    artifact_path.split("/")[-1] if "/" in artifact_path else artifact_path
                ]

                for path in possible_paths:
                    try:
                        artifacts = client.list_artifacts(run_id, path)
                        csv_files = [a for a in artifacts if a.path.endswith('.csv')]

                        if csv_files:
                            artifact_uri = f"runs:/{run_id}/{csv_files[0].path}"
                            local_path = mlflow.artifacts.download_artifacts(artifact_uri)
                            return pd.read_csv(local_path)
                    except Exception:
                        continue

            except Exception:
                continue

        return None

    except Exception as e:
        print(f"[load_latest_artifact_dataframe] failed: {e}")
        return None