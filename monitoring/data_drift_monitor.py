# monitoring/data_drift_monitor.py
"""
Data Drift 모니터링: 훈련 데이터와 프로덕션 데이터 분포 비교
"""
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
import mlflow
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """데이터 드리프트 모니터링 클래스"""

    def __init__(self, reference_data: pd.DataFrame, feature_columns: list):
        """
        Args:
            reference_data: 훈련 데이터 (기준 데이터)
            feature_columns: 모니터링할 피처 컬럼 리스트
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns

        # 컬럼 매핑 (타겟이 있는 경우)
        self.column_mapping = ColumnMapping(
            numerical_features=feature_columns,
            target=None  # 예측 단계에서는 타겟이 없음
        )

    def detect_drift(self, current_data: pd.DataFrame) -> dict:
        """드리프트 탐지"""
        logger.info("Detecting data drift...")

        # Evidently Report 생성
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])

        # 리포트 실행
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        # 결과 추출
        drift_results = report.as_dict()

        # 주요 메트릭 추출
        dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']
        n_drifted_columns = drift_results['metrics'][0]['result']['number_of_drifted_columns']

        logger.info(f"Dataset drift detected: {dataset_drift}")
        logger.info(f"Number of drifted columns: {n_drifted_columns}")

        return {
            'dataset_drift': dataset_drift,
            'n_drifted_columns': n_drifted_columns,
            'drift_results': drift_results,
            'report': report
        }

    def save_report(self, report: Report, output_dir: Path = None):
        """HTML 리포트 저장"""
        if output_dir is None:
            output_dir = Path("/workspace/monitoring/reports")

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"drift_report_{timestamp}.html"

        report.save_html(str(report_path))
        logger.info(f"Drift report saved: {report_path}")

        return report_path

    def log_to_mlflow(self, drift_results: dict, report_path: Path):
        """MLflow에 드리프트 결과 로깅"""
        with mlflow.start_run(run_name=f"drift_check_{datetime.now().strftime('%Y%m%d')}"):
            # 메트릭 로깅
            mlflow.log_metric("dataset_drift", int(drift_results['dataset_drift']))
            mlflow.log_metric("n_drifted_columns", drift_results['n_drifted_columns'])

            # 파라미터 로깅
            mlflow.log_param("check_timestamp", datetime.now().isoformat())
            mlflow.log_param("n_features", len(self.feature_columns))

            # 리포트 아티팩트 로깅
            mlflow.log_artifact(str(report_path))

            logger.info("Drift results logged to MLflow")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Data drift monitoring")
    parser.add_argument("--reference-data", type=str, required=True, help="Path to reference (training) data")
    parser.add_argument("--current-data", type=str, required=True, help="Path to current (production) data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")

    args = parser.parse_args()

    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("data_drift_monitoring")

    # 데이터 로드
    logger.info(f"Loading reference data: {args.reference_data}")
    reference_df = pd.read_csv(args.reference_data)

    logger.info(f"Loading current data: {args.current_data}")
    current_df = pd.read_csv(args.current_data)

    # 피처 컬럼 추출 (날짜, 타겟 제외)
    feature_columns = [col for col in reference_df.columns
                       if col not in ['date', 'new_cases', 'y_next']]

    # 드리프트 모니터링 초기화
    monitor = DataDriftMonitor(reference_df, feature_columns)

    # 드리프트 탐지
    drift_results = monitor.detect_drift(current_df)

    # 리포트 저장
    report_path = monitor.save_report(
        drift_results['report'],
        Path(args.output_dir) if args.output_dir else None
    )

    # MLflow에 로깅
    monitor.log_to_mlflow(drift_results, report_path)

    # 경고 발생
    if drift_results['dataset_drift']:
        logger.warning(f"⚠️ DATA DRIFT DETECTED! {drift_results['n_drifted_columns']} columns drifted.")
        logger.warning("Consider retraining the model with recent data.")
    else:
        logger.info("✅ No significant data drift detected.")


if __name__ == "__main__":
    import os

    main()