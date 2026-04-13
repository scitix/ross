import json
import os
import math
import time
import numpy as np
import xgboost as xgb
import joblib

import logging
logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Tuple
from dataclasses import asdict
from sklearn.preprocessing import StandardScaler


from common.models import BaseModel
from common.config import InferenceConfig
from common.features import PlatformPerf, RegressionFeatures, SGLRegressionFeatures

# from scheduler_impl import SchedulerOutputForRegression

class ROSSModel:
    """
    Loads a pre-trained performance prediction model (XGBoost) and provides an 
    interface to predict iteration time based on scheduler output.
    """
    def __init__(self,
                saved_model_path: str,
                platform_perf: PlatformPerf,
                model: BaseModel,
                inference_config: InferenceConfig,
                regressor: str
                ):
        """
        Initializes the ROSSModel instance.

        Args:
            saved_model_path (str): Path to the saved XGBoost model file (e.g., 'model.json').
            platform_perf (PlatformPerf): An object containing platform performance metrics.
            model (BaseModel): An object containing the base model architecture information 
                                (e.g., num_layers, num_heads).
            inference_config (InferenceConfig): An object containing the model's parallel configuration 
                                        (TP, PP, etc.).
        """
        if regressor not in ["xgboost", "linear"]:
            raise ValueError(f"Unsupported regressor type: '{regressor}'. Use 'xgboost' or 'linear'.")
        
        self.regressor_type = regressor
        self.saved_model_path = saved_model_path

        self.model = model
        self.inference_config = inference_config
        self.platform_perf = platform_perf
        
        self.regression_model = self._load_model()
        with open(os.path.join(self.saved_model_path, "model_metadata.json"), 'r') as f:
            self.regression_metadata = json.load(f)

    def _load_model(self) -> Any:
        """Loads the regression model from disk based on the specified type."""
        if self.regressor_type == "xgboost":
            model_file = os.path.join(self.saved_model_path, "model.json")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"XGBoost model file not found at: {model_file}")
            model = xgb.XGBRegressor()
            model.load_model(model_file)
            return model
        
        elif self.regressor_type == "linear":
            model_file = os.path.join(self.saved_model_path, "model.joblib")
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Linear Regression model file not found at: {model_file}")
            model = joblib.load(model_file)
            return model
    
    def _extract_features(self,
                        req_ids: List[str],
                        prefill_seq_lens: List[int],
                        decode_seq_lens: List[int]) -> RegressionFeatures:
        """
        Extracts raw data from a SchedulerOutput object, initializes the RegressionFeatures
        class to calculate all derived features, and finally converts the result into a dictionary.

        Args:
            sched_output (SchedulerOutput): The output of a single scheduling decision from the scheduler,
                                            containing information about the prefill and decode stages.

        Returns:
            RegressionFeatures: A dictionary containing all the calculated features and their values,
                                to be used for model prediction.
        """
        feature_vals = RegressionFeatures(
            batch_size=len(req_ids),
            prefill_seq_lens=prefill_seq_lens,
            decode_seq_lens=decode_seq_lens,

            model=self.model,
            inference_config=self.inference_config,
            platform_perf=self.platform_perf,
        )
        return feature_vals

    def _run_prediction(self, feature_vals, extra: dict = None) -> float:
        """Build input vector, run the regressor, and return the clamped result.

        Args:
            feature_vals: A features dataclass with a `.platform_perf` attribute.
            extra: Optional mapping of feature-column-name → scalar value for any
                   columns not present on *feature_vals* (e.g. ``{"isl": 512, "osl": 128}``).
        """
        input_vector = []
        feature_dict = {}
        extra = extra or {}
        for col in self.regression_metadata["feature_names"]:
            if col.startswith("platform_") or col.startswith("theoretical_"):
                val = getattr(feature_vals.platform_perf, col)
            elif col in extra:
                val = extra[col]
            else:
                val = getattr(feature_vals, col)
            feature_dict[col] = val
            input_vector.append(val)

        input_array = np.array(input_vector).reshape(1, -1)
        if self.regression_metadata["scale_needed"]:
            scaler = StandardScaler()
            input_array = scaler.fit_transform(input_array)

        prediction = self.regression_model.predict(input_array)
        return max(float(prediction[0]), 0)

    def predict(self, req_ids: List[str],
                    prefill_seq_lens: List[int],
                    decode_seq_lens: List[int],
                    isl: int,
                    osl: int) -> float:
        """
        Predicts the iteration time for a given schedule_output.

        Args:
            schedule_output: The output from the scheduler's schedule() method.

        Returns:
            A tuple of (time_step, forward_overhead).
        """
        if not req_ids:
            return 0
        feature_vals = self._extract_features(req_ids, prefill_seq_lens, decode_seq_lens)
        return self._run_prediction(feature_vals, extra={"isl": isl, "osl": osl})


class SGLROSSModel(ROSSModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_predict_stats()

    def reset_predict_stats(self) -> None:
        self.predict_call_count = 0
        self.predict_total_time_s = 0.0
        self.predict_feature_time_s = 0.0
        self.predict_model_time_s = 0.0

    def get_predict_stats(self) -> Dict[str, float]:
        avg_time_ms = 0.0
        if self.predict_call_count > 0:
            avg_time_ms = self.predict_total_time_s * 1000.0 / self.predict_call_count
        feature_avg_time_ms = 0.0
        model_avg_time_ms = 0.0
        if self.predict_call_count > 0:
            feature_avg_time_ms = self.predict_feature_time_s * 1000.0 / self.predict_call_count
            model_avg_time_ms = self.predict_model_time_s * 1000.0 / self.predict_call_count
        return {
            "predict_call_count": self.predict_call_count,
            "predict_total_time_s": self.predict_total_time_s,
            "predict_avg_time_ms": avg_time_ms,
            "predict_feature_time_s": self.predict_feature_time_s,
            "predict_feature_avg_time_ms": feature_avg_time_ms,
            "predict_model_time_s": self.predict_model_time_s,
            "predict_model_avg_time_ms": model_avg_time_ms,
        }

    def _extract_features(self,
                          req_ids: List[str],
                          seq_lens: List[int]) -> SGLRegressionFeatures:
        return SGLRegressionFeatures(
            batch_size=len(req_ids),
            seq_lens=seq_lens,
            model=self.model,
            inference_config=self.inference_config,
            platform_perf=self.platform_perf,
        )

    def predict(self, req_ids: List[str], seq_lens: List[int]) -> float:
        if not req_ids:
            return 0
        start_t = time.perf_counter()
        feature_start_t = start_t
        feature_vals = self._extract_features(req_ids, seq_lens)
        feature_end_t = time.perf_counter()
        model_start_t = feature_end_t
        prediction = self._run_prediction(feature_vals)
        model_end_t = time.perf_counter()
        self.predict_call_count += 1
        self.predict_feature_time_s += feature_end_t - feature_start_t
        self.predict_model_time_s += model_end_t - model_start_t
        self.predict_total_time_s += model_end_t - start_t
        return prediction
