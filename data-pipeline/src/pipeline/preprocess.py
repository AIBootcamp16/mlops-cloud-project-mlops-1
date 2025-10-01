# data-pipeline/src/pipeline/preprocess.py
# -*- coding: utf-8 -*-
"""
Preprocessing task with enhanced error handling and validation
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="preprocessing")
    ap.add_argument("--from-latest", action="store_true")
    ap.add_argument("--tracking-uri", type=str, default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("PREPROCESSING TASK START")
    print("=" * 60)
    print(f"Run name: {args.run_name}")
    print(f"From latest: {args.from_latest}")
    print(f"Tracking URI: {args.tracking_uri or 'default'}")
    print("=" * 60)

    try:
        from src.data_processing.data_preprocessor import CovidDataPreprocessor
        pre = CovidDataPreprocessor(tracking_uri=args.tracking_uri)

        result = pre.run(
            run_name=args.run_name,
            from_latest=args.from_latest
        )

        print("\n" + "=" * 60)
        print("✅ PREPROCESSING SUCCESS")
        print("=" * 60)
        print(f"Output shape: {result.shape if result is not None else 'N/A'}")

        if result is not None and 'date' in result.columns:
            print(f"Date range: {result['date'].min().date()} ~ {result['date'].max().date()}")

        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ PREPROCESSING FAILED")
        print("=" * 60)
        print(f"Error: {e}")

        import traceback
        traceback.print_exc()

        print("=" * 60)
        raise


if __name__ == "__main__":
    main()