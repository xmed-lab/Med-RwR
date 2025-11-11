# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
import os
import sys

from swift.utils.import_utils import _LazyModule

_extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}

# Get engine type from environment variable
engine_type = os.getenv('SWIFT_ENGINE_TYPE', 'pt_engine_retrieve_infer')

# Define all possible engine mappings
_all_engines = {
    'vllm_engine': ['VllmEngine'],
    'grpo_vllm_engine': ['GRPOVllmEngine'],
    'lmdeploy_engine': ['LmdeployEngine'],
    'pt_engine_retrieve_infer': ['PtEngine'],
    'pt_engine_retrieve_infer_img': ['PtEngine'],
    'pt_engine_retrieve_train': ['PtEngine'],
    'pt_engine': ['PtEngine'],
}

# Build the import structure based on environment variable
_import_structure = {
    'vllm_engine': ['VllmEngine'],
    'grpo_vllm_engine': ['GRPOVllmEngine'],
    'lmdeploy_engine': ['LmdeployEngine'],
    'infer_client': ['InferClient'],
    'infer_engine': ['InferEngine'],
    'base': ['BaseInferEngine'],
    'utils': ['prepare_generation_config', 'AdapterRequest', 'set_device_context'],
}

# Add the selected engine
if engine_type in _all_engines:
    _import_structure[engine_type] = _all_engines[engine_type]
else:
    # Fallback to default
    _import_structure['pt_engine_retrieve_infer'] = ['PtEngine']

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
    extra_objects={},
)