"""
예측 결과 저장 스크립트: 로컬 또는 S3에 저장
"""
import argparse
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_prediction(predictions_dir: Path = None) -> Path:
    """가장 최근 예측 결과 파일 찾기"""
    if predictions_dir is None:
        predictions_dir = Path("/workspace/data/predictions")

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    # CSV 파일 찾기
    prediction_files = list(predictions_dir.glob("batch_prediction_*.csv"))

    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {predictions_dir}")

    # 가장 최근 파일 선택
    latest_file = max(prediction_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest prediction file: {latest_file}")

    return latest_file


def save_to_local(df: pd.DataFrame, output_path: str):
    """로컬에 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"✅ Predictions saved to local: {output_path}")

    return str(output_path)


def save_to_s3(df: pd.DataFrame, s3_path: str):
    """S3에 저장"""
    try:
        import boto3
        from botocore.exceptions import ClientError

        # S3 경로 파싱
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")

        s3_path = s3_path.replace("s3://", "")
        bucket_name = s3_path.split("/")[0]
        key = "/".join(s3_path.split("/")[1:])

        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not key.endswith(".csv"):
            key = f"{key}/batch_prediction_{timestamp}.csv"

        logger.info(f"Uploading to S3: s3://{bucket_name}/{key}")

        # CSV를 메모리에서 직접 업로드
        csv_buffer = df.to_csv(index=False)

        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=csv_buffer.encode('utf-8'),
            ContentType='text/csv'
        )

        logger.info(f"✅ Predictions saved to S3: s3://{bucket_name}/{key}")
        return f"s3://{bucket_name}/{key}"

    except ImportError:
        logger.error("❌ boto3 not installed. Install with: pip install boto3")
        raise
    except ClientError as e:
        logger.error(f"❌ S3 upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Save batch predictions")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input prediction file path (default: find latest in /workspace/data/predictions)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (local path or s3://bucket/path/)"
    )

    args = parser.parse_args()

    # 1. 입력 파일 찾기
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_prediction()

    logger.info(f"Loading predictions from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} predictions")

    # 2. 저장 위치 결정
    output_path = args.output

    if output_path.startswith("s3://"):
        # S3에 저장
        saved_path = save_to_s3(df, output_path)
    else:
        # 로컬에 저장
        # 타임스탬프 추가
        if output_path.endswith("/"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_path}batch_prediction_{timestamp}.csv"

        saved_path = save_to_local(df, output_path)

    logger.info(f"🎉 Predictions successfully saved to: {saved_path}")

    # 3. 통계 정보 출력
    if 'predicted_new_cases' in df.columns:
        logger.info("📊 Prediction Statistics:")
        logger.info(f"  - Count: {len(df)}")
        logger.info(f"  - Mean: {df['predicted_new_cases'].mean():.2f}")
        logger.info(f"  - Max: {df['predicted_new_cases'].max():.2f}")
        logger.info(f"  - Min: {df['predicted_new_cases'].min():.2f}")


if __name__ == "__main__":
    main()