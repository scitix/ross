from typing import Optional

import sgl_sim.bench_sglang
import vllm_sim.bench_vllm

from pareto.inference_summary import InferenceSummary
from common.config import InferenceConfig, RuntimeConfig

BACKEND_MODULES = {
    'sglang': sgl_sim.bench_sglang,
    'vllm':   vllm_sim.bench_vllm,
}


class InferenceSession:
    """Thin wrapper around backend bench modules.

    Supports both colocated and disaggregated parallelism through a single
    ``find_best_result_under_constraints`` method.  Pass a single
    ``inference_config`` for colocated runs, or ``prefill_inference_config``
    and ``decode_inference_config`` for disaggregated runs.
    """

    def __init__(self, model_uri: str, backend: str) -> None:
        self._model_uri = model_uri
        self._backend   = BACKEND_MODULES[backend]

    def find_best_result_under_constraints(
        self,
        runtime_config: RuntimeConfig,
        gpu: str,
        top_k: int = 1,
        inference_config: Optional[InferenceConfig] = None,
        prefill_inference_config: Optional[InferenceConfig] = None,
        decode_inference_config:  Optional[InferenceConfig] = None,
    ) -> InferenceSummary:
        is_disagg = prefill_inference_config is not None

        if is_disagg:
            return self._backend.find_best_disagg_result_under_constraints(
                self._model_uri,
                prefill_inference_config,
                decode_inference_config,
                runtime_config,
                top_k=top_k,
                gpu=gpu,
            )
        else:
            return self._backend.find_best_colocate_result_under_constraints(
                self._model_uri,
                inference_config,
                runtime_config,
                top_k=top_k,
                gpu=gpu,
            )
