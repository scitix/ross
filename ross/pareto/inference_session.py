import json
import optparse
import sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from pareto.inference_summary import InferenceSummary
from common.config import InferenceConfig, RuntimeConfig
from common.models import BaseModel

import sglang.bench_sglang
import vllm_sim.bench_vllm
BACKEND_MODULES = {
    'sglang': sglang.bench_sglang,
    'vllm': vllm_sim.bench_vllm,
}

class InferenceSession:
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (BaseModel): the model to run inference
        backend (backend.Backend): the backend to run inference

    Methods:
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary
            which contains all the possible agg config and perf that matchs SLA.
    """

    def __init__(self, model_uri: str,  backend: str) -> None:
        """
        Initialize the InferenceSession
        """
        self._model_uri = model_uri
        self._backend = BACKEND_MODULES[backend]

    def find_best_result_under_constraints(
        self, 
        inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        gpu: str,
        top_k: int = 1,
    ) -> InferenceSummary:
        return self._backend.find_best_colocate_result_under_constraints(
            self._model_uri, inference_config, runtime_config, top_k=top_k, gpu=gpu,
        )

class DisaggInferenceSession:
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (BaseModel): the model to run inference
        backend (backend.Backend): the backend to run inference

    Methods:
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary
            which contains all the possible agg config and perf that matchs SLA.
    """

    def __init__(self, model_uri: str,  backend: str) -> None:
        """
        Initialize the InferenceSession
        """
        self._model_uri = model_uri
        self._backend = BACKEND_MODULES[backend]

    def find_best_result_under_constraints(
        self, 
        prefill_inference_config: InferenceConfig,
        decode_inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        gpu: str,
        top_k: int = 1,
    ) -> InferenceSummary:
        return self._backend.find_best_disagg_result_under_constraints(
            self._model_uri, prefill_inference_config, decode_inference_config, runtime_config, top_k=top_k, gpu=gpu,
        )
