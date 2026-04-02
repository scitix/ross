import json
import os
import math
import numpy as np
import joblib

from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

from common.features import SGLRegressionFeatures
from common.ross_model import ROSSModel

class SGLROSSModel(ROSSModel):
    def _extract_features(self,
                        req_ids: List[str],
                        seq_lens: List[int]) -> SGLRegressionFeatures:

        feature_vals = SGLRegressionFeatures(
            batch_size=len(req_ids),
            seq_lens=seq_lens,
            model=self.model,
            inference_config=self.inference_config,
            platform_perf=self.platform_perf,
        )
        return feature_vals

    def predict(self, req_ids: List[str],
                seq_lens: List[int]) -> float:

        if not req_ids:
            return 0
        # Step 1: Extract features from the scheduler output
        input_vector = []
        feature_dict = {}
        feature_vals = self._extract_features(req_ids, seq_lens)
        for col in self.regression_metadata["feature_names"]:
            if col.startswith("platform_") or col.startswith('theoretical_'):
                feature_dict[col] = getattr(feature_vals.platform_perf, col)
                input_vector.append(getattr(feature_vals.platform_perf, col))
            else:
                feature_dict[col] = getattr(feature_vals, col)
                input_vector.append(getattr(feature_vals, col))
        # print(feature_dict, indent=4)
        # a = input()
        input_array = np.array(input_vector).reshape(1, -1)

        # check if scale needed
        if self.regression_metadata["scale_needed"] == True:
            scaler = StandardScaler()
            input_array = scaler.fit_transform(input_array)
        # Step 4: Perform prediction
        prediction = self.regression_model.predict(input_array)
        
        return max(float(prediction[0]), 0)
