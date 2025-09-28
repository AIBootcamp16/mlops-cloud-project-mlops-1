# src/feature_engineering/feature_engineer.py - 강화된 버전
import re
import os
import tempfile
import json  # 추가
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler
import mlflow
import mlflow.data
import warnings

warnings.filterwarnings('ignore')

from ..config.settings import ProjectConfig
from ..utils.mlflow_utils import MLflowManager


def _sanitize_mlflow_name(name: str) -> str:
    """
    MLflow metric/param 이름에서 허용되지 않는 문자를 '_'로 치환.
    허용: 알파벳, 숫자, '_', '-', '.', ' ', ':', '/'
    """
    return re.sub(r"[^a-zA-Z0-9_\-.:/ ]", "_", name)


class CovidFeatureEngineer:
    """MLflow 통합 피처 엔지니어링 - 강화된 버전"""

    def __init__(self, config: Optional[ProjectConfig] = None, tracking_uri: Optional[str] = None):
        self.config = config or ProjectConfig(tracking_uri=tracking_uri)
        if tracking_uri:
            self.config.mlflow.TRACKING_URI = tracking_uri

        self.mlflow_manager = MLflowManager(
            self.config.mlflow.TRACKING_URI,
            "covid_feature_engineering"
        )
        self.scaler = None

    def engineer_features(self, processed_data: pd.DataFrame,
                          target_column: str = 'new_cases',
                          run_name: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """피처 엔지니어링 실행"""

        with self.mlflow_manager.start_run(run_name):
            print("Starting MLflow-tracked feature engineering...")

            # 파라미터 로깅
            self._log_feature_params(processed_data, target_column)

            # 피처 엔지니어링 파이프라인 실행
            features_df, target_series = self._execute_feature_pipeline(processed_data, target_column)

            # 결과 로깅
            self._log_feature_results(processed_data, features_df, target_series)

            print(f"Feature engineering completed! Check MLflow UI: {self.config.mlflow.TRACKING_URI}")
            return features_df, target_series

    def _log_feature_params(self, data: pd.DataFrame, target_column: str) -> None:
        """피처 엔지니어링 파라미터 로깅"""
        mlflow.log_param("input_shape", f"{data.shape[0]}x{data.shape[1]}")
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("feature_engineering_timestamp", datetime.now().isoformat())
        mlflow.log_param("lag_features_days", [1, 3, 7, 14])
        mlflow.log_param("rolling_windows", [7, 14, 30])
        mlflow.log_param("scaling_method", "robust")  # StandardScaler 대신 RobustScaler 사용

    def _execute_feature_pipeline(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """피처 엔지니어링 파이프라인 실행"""

        # 데이터 정렬 (날짜순)
        data = data.sort_values('date').reset_index(drop=True)

        # 초기 데이터 정제
        print("Step 0: Initial data cleaning...")
        data = self._initial_cleaning(data)

        # 1단계: 시계열 피처 생성
        print("Step 1: Creating time series features...")
        features_df = self._create_time_series_features(data.copy())
        features_df = self._clean_step(features_df, "after time series features")
        mlflow.log_metric("step1_features", len(features_df.columns))

        # 2단계: 래그 피처 생성
        print("Step 2: Creating lag features...")
        features_df = self._create_lag_features(features_df, target_column)
        features_df = self._clean_step(features_df, "after lag features")
        mlflow.log_metric("step2_features", len(features_df.columns))

        # 3단계: 이동평균 피처 생성
        print("Step 3: Creating rolling window features...")
        features_df = self._create_rolling_features(features_df, target_column)
        features_df = self._clean_step(features_df, "after rolling features")
        mlflow.log_metric("step3_features", len(features_df.columns))

        # 4단계: 통계적 피처 생성
        print("Step 4: Creating statistical features...")
        features_df = self._create_statistical_features(features_df, target_column)
        features_df = self._clean_step(features_df, "after statistical features")
        mlflow.log_metric("step4_features", len(features_df.columns))

        # 5단계: 최종 데이터 정제
        print("Step 5: Final data cleaning...")
        features_df = self._final_cleaning(features_df)

        # 6단계: 스케일링
        print("Step 6: Scaling features...")
        features_df = self._scale_features_safe(features_df)

        # 타겟 변수 분리
        if target_column not in features_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")

        target_series = features_df[target_column].copy()
        features_df = features_df.drop(columns=[target_column])

        # 결측치 제거 (래그 피처로 인한)
        print("Step 7: Removing rows with missing values...")
        valid_mask = features_df.notna().all(axis=1) & target_series.notna() & np.isfinite(target_series)
        features_df = features_df[valid_mask].reset_index(drop=True)
        target_series = target_series[valid_mask].reset_index(drop=True)

        mlflow.log_metric("final_samples", len(features_df))
        mlflow.log_metric("final_features", len(features_df.columns))

        return features_df, target_series

    def _initial_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """초기 데이터 정제"""
        print("  - Initial cleaning: replacing inf/nan values...")

        # 수치형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in data.columns:
                # 무한값을 NaN으로 변환
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)

                # 매우 큰 값 클리핑
                if data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    q99 = data[col].quantile(0.99)
                    q01 = data[col].quantile(0.01)

                    if pd.notna(q99) and pd.notna(q01):
                        data[col] = data[col].clip(lower=q01, upper=q99)

        return data

    def _clean_step(self, data: pd.DataFrame, step_name: str) -> pd.DataFrame:
        """각 단계별 데이터 정제"""
        print(f"  - Cleaning {step_name}...")

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in data.columns:
                # 무한값 처리
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)

                # 매우 큰 값 처리 (더 보수적으로)
                if data[col].dtype in ['float64', 'float32']:
                    # 절댓값이 1e6 이상인 값들을 클리핑
                    mask_large = np.abs(data[col]) > 1e6
                    if mask_large.any():
                        median_val = data[col].median()
                        data.loc[mask_large, col] = median_val

        return data

    def _final_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """최종 데이터 정제"""
        print("  - Final cleaning: comprehensive data validation...")

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in data.columns:
                # 1. 무한값 제거
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)

                # 2. 매우 큰 값 제거 (더 엄격하게)
                abs_vals = np.abs(data[col])
                mask_large = abs_vals > 1e3  # 1000 이상의 값
                if mask_large.any():
                    median_val = data[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    data.loc[mask_large, col] = median_val
                    print(f"    - Capped {mask_large.sum()} large values in {col}")

                # 3. NaN 값 처리
                if data[col].isnull().any():
                    # Forward fill -> Backward fill -> 0으로 채우기
                    data[col] = data[col].ffill().bfill().fillna(0)

        # 최종 검증
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        nan_count = data.isnull().sum().sum()

        print(f"  - After final cleaning: {inf_count} inf values, {nan_count} NaN values")

        # 여전히 문제가 있다면 강제로 0으로 설정
        if inf_count > 0 or nan_count > 0:
            print("  - Force cleaning remaining problematic values...")
            data = data.fillna(0)
            data = data.replace([np.inf, -np.inf], 0)

        # ✅ datetime 컬럼 제거 (모델 입력에는 불필요)
        datetime_cols = data.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        if len(datetime_cols) > 0:
            data = data.drop(columns=list(datetime_cols))
            print(f"  - Dropped datetime columns: {list(datetime_cols)}")

        return data

    def _create_time_series_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시계열 관련 피처 생성"""
        try:
            data['quarter'] = data['date'].dt.quarter
            data['day_of_year'] = data['date'].dt.dayofyear
            data['week_of_year'] = data['date'].dt.isocalendar().week

            # 순환 피처 (주기성 포착)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        except Exception as e:
            print(f"  - Warning in time series features: {e}")

        return data

    def _create_lag_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """래그 피처 생성"""
        lag_days = [1, 3, 7, 14]
        lag_columns = ['new_cases', 'new_deaths', 'total_cases']

        for col in lag_columns:
            if col in data.columns:
                try:
                    for lag in lag_days:
                        lag_col_name = f'{col}_lag_{lag}'
                        data[lag_col_name] = data[col].shift(lag)
                        # 즉시 무한값 처리
                        data[lag_col_name] = data[lag_col_name].replace([np.inf, -np.inf], np.nan)
                except Exception as e:
                    print(f"  - Warning in lag features for {col}: {e}")

        return data

    def _create_rolling_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """이동평균 피처 생성"""
        windows = [7, 14, 30]
        rolling_columns = ['new_cases', 'new_deaths', 'stringency_index', 'reproduction_rate']

        for col in rolling_columns:
            if col in data.columns:
                try:
                    for window in windows:
                        # 이동평균
                        data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window, min_periods=1).mean()
                        # 이동표준편차 (0으로 나누기 방지)
                        rolling_std = data[col].rolling(window=window, min_periods=1).std()
                        data[f'{col}_rolling_std_{window}'] = rolling_std.fillna(0)
                        # 최대값/최소값
                        data[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window, min_periods=1).max()
                        data[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window, min_periods=1).min()

                        # 각 피처별로 즉시 무한값 처리
                        for suffix in ['_rolling_mean_', '_rolling_std_', '_rolling_max_', '_rolling_min_']:
                            feature_name = f'{col}{suffix}{window}'
                            if feature_name in data.columns:
                                data[feature_name] = data[feature_name].replace([np.inf, -np.inf], np.nan)

                except Exception as e:
                    print(f"  - Warning in rolling features for {col}: {e}")

        return data

    def _create_statistical_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """통계적 피처 생성"""
        # 안전한 변화율 계산
        change_columns = ['total_cases', 'total_deaths']
        for col in change_columns:
            if col in data.columns:
                try:
                    # 퍼센트 변화율 (더 안전한 방법)
                    pct_change = data[col].pct_change()
                    # 무한값과 매우 큰 값 처리
                    pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                    pct_change = pct_change.clip(lower=-1, upper=1)  # ±100% 로 제한
                    data[f'{col}_pct_change'] = pct_change.fillna(0)

                    # 절대 차이
                    diff_val = data[col].diff().fillna(0)
                    data[f'{col}_diff'] = diff_val.replace([np.inf, -np.inf], 0)

                except Exception as e:
                    print(f"  - Warning in statistical features for {col}: {e}")

        # 매우 안전한 비율 피처
        try:
            if 'new_deaths' in data.columns and 'new_cases' in data.columns:
                # 분모를 더 크게 하여 0 나눗셈 완전 방지
                denominator = np.maximum(data['new_cases'], 1.0)  # 최소값 1
                data['death_rate'] = data['new_deaths'] / denominator
                data['death_rate'] = data['death_rate'].clip(lower=0, upper=1)  # 0-100% 제한

            if 'icu_patients' in data.columns and 'new_cases' in data.columns:
                denominator = np.maximum(data['new_cases'], 1.0)
                data['icu_rate'] = data['icu_patients'] / denominator
                data['icu_rate'] = data['icu_rate'].clip(lower=0, upper=10)  # 상식적 범위

        except Exception as e:
            print(f"  - Warning in ratio features: {e}")

        # 계절성 지표 (안전함)
        try:
            data['is_summer'] = ((data['month'] >= 6) & (data['month'] <= 8)).astype(int)
            data['is_winter'] = ((data['month'] <= 2) | (data['month'] == 12)).astype(int)
        except Exception as e:
            print(f"  - Warning in seasonal features: {e}")

        return data

    def _scale_features_safe(self, data: pd.DataFrame) -> pd.DataFrame:
        """안전한 피처 스케일링"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        non_target_numeric = [col for col in numeric_columns if col not in ['new_cases', 'date']]

        if non_target_numeric:
            print(f"  - Scaling {len(non_target_numeric)} numeric features...")

            # 스케일링할 데이터 추출
            scaling_data = data[non_target_numeric].copy()

            # 최종 데이터 검증
            print("  - Pre-scaling validation...")
            inf_mask = np.isinf(scaling_data)
            nan_mask = np.isnan(scaling_data)
            large_mask = np.abs(scaling_data) > 100

            print(f"    - Inf values: {inf_mask.sum().sum()}")
            print(f"    - NaN values: {nan_mask.sum().sum()}")
            print(f"    - Large values (>100): {large_mask.sum().sum()}")

            # 문제 있는 값들을 0으로 대체
            scaling_data = scaling_data.fillna(0)
            scaling_data = scaling_data.replace([np.inf, -np.inf], 0)

            # RobustScaler 사용 (이상치에 더 강함)
            try:
                self.scaler = RobustScaler()
                scaled_data = self.scaler.fit_transform(scaling_data)

                # 스케일링 결과 검증
                if np.isfinite(scaled_data).all():
                    data[non_target_numeric] = scaled_data
                    print("  - Scaling completed successfully with RobustScaler")
                else:
                    # 수동 정규화
                    print("  - RobustScaler failed, using manual normalization...")
                    for col in non_target_numeric:
                        col_data = scaling_data[col]
                        if col_data.std() > 0:
                            data[col] = (col_data - col_data.mean()) / col_data.std()
                        else:
                            data[col] = 0

            except Exception as e:
                print(f"  - Scaling error: {e}")
                print("  - Using min-max normalization as fallback...")

                # 최종 대안: 간단한 min-max 정규화
                for col in non_target_numeric:
                    col_data = scaling_data[col]
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max > col_min:
                        data[col] = (col_data - col_min) / (col_max - col_min)
                    else:
                        data[col] = 0

            # 스케일러 저장 (성공한 경우만)
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    import joblib
                    scaler_filename = "feature_scaler.joblib"
                    joblib.dump(self.scaler, scaler_filename)
                    mlflow.log_artifact(scaler_filename, "scalers")
                    import os
                    os.remove(scaler_filename)
                except:
                    pass

        return data

    def _log_feature_results(self, input_data: pd.DataFrame, features_df: pd.DataFrame,
                             target_series: pd.Series) -> None:
        """피처 엔지니어링 결과 로깅"""

        # 기본 메트릭
        mlflow.log_metric("input_samples", len(input_data))
        mlflow.log_metric("output_samples", len(features_df))
        mlflow.log_metric("input_features", len(input_data.columns))
        mlflow.log_metric("output_features", len(features_df.columns))
        mlflow.log_metric("feature_expansion_ratio", len(features_df.columns) / len(input_data.columns))

        # 타겟 변수 통계
        mlflow.log_metric("target_mean", float(target_series.mean()))
        mlflow.log_metric("target_std", float(target_series.std()))
        mlflow.log_metric("target_min", float(target_series.min()))
        mlflow.log_metric("target_max", float(target_series.max()))

        mlflow.log_metric("rows_input", len(input_data))
        mlflow.log_metric("rows_features", len(features_df))
        mlflow.log_metric("features_total", features_df.shape[1])

        # dtype 별 feature count 로깅 (패치된 부분)
        feature_types = features_df.dtypes.value_counts().to_dict()
        for dtype, count in feature_types.items():
            dtype_clean = _sanitize_mlflow_name(str(dtype))
            mlflow.log_metric(f"features_{dtype_clean}_count", count)

        # 데이터셋 등록 (수정)
        tmp_df = features_df.copy()
        target_col_name = target_series.name or "new_cases"  # 시리즈에 name 없으면 기본값
        tmp_df[target_col_name] = target_series.values

        dataset = mlflow.data.from_pandas(
            tmp_df,
            targets=target_col_name,
            name="korea_covid_features_dataset"
        )
        mlflow.log_input(dataset, context="feature_engineering")

        # 피처와 타겟을 함께 저장 (임시 디렉터리 사용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = tempfile.gettempdir()  # /tmp 사용

        try:
            # 피처 데이터 저장
            features_filename = f"covid_features_{timestamp}.csv"
            features_path = os.path.join(tmp_dir, features_filename)
            features_df.to_csv(features_path, index=False)
            mlflow.log_artifact(features_path, "features")

            # 타겟 데이터 저장
            target_filename = f"covid_target_{timestamp}.csv"
            target_path = os.path.join(tmp_dir, target_filename)
            target_series.to_csv(target_path, index=False, header=['target'])
            mlflow.log_artifact(target_path, "targets")

            # 피처 이름 저장
            feature_names = list(features_df.columns)
            feature_names_filename = f"feature_names_{timestamp}.json"
            feature_names_path = os.path.join(tmp_dir, feature_names_filename)
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
            mlflow.log_artifact(feature_names_path, "metadata")

            print(f"Feature engineering results logged:")
            print(f"  - Features: features/{features_filename}")
            print(f"  - Targets: targets/{target_filename}")
            print(f"  - Feature names: metadata/{feature_names_filename}")

        except Exception as e:
            print(f"[_log_feature_results] File save error: {e}")
            # 아티팩트 저장 실패해도 메트릭은 이미 로깅되었으므로 계속 진행

        finally:
            # 임시 파일 정리
            for filepath in [features_path, target_path, feature_names_path]:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except Exception:
                    pass  # 정리 실패해도 무시