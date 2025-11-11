# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING
import os
import sys

from swift.utils.import_utils import _LazyModule

_extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}

# Get trainer type from environment variable
trainer_type = os.getenv('SWIFT_GRPO_TRAINER_TYPE', 'grpo_trainer_retrieve_new')

# Define all possible GRPO trainer mappings
_all_grpo_trainers = {
    'grpo_trainer': ['GRPOTrainer'],
    'grpo_trainer_retrieve': ['GRPOTrainer'],
}

# Build the import structure with all non-GRPO trainers
_import_structure = {
    'cpo_trainer': ['CPOTrainer'],
    'dpo_trainer': ['DPOTrainer'],
    'kto_trainer': ['KTOTrainer'],
    'orpo_trainer': ['ORPOTrainer'],
    'ppo_trainer': ['PPOTrainer'],
    'reward_trainer': ['RewardTrainer'],
    'rlhf_mixin': ['RLHFTrainerMixin'],
}

# Add the selected GRPO trainer (only one will be included)
if trainer_type in _all_grpo_trainers:
    _import_structure[trainer_type] = _all_grpo_trainers[trainer_type]
else:
    # Fallback to default
    _import_structure['grpo_trainer_retrieve'] = ['GRPOTrainer']

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
    extra_objects={},
)
