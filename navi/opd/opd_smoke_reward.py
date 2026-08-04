# Copyright 2025 Navi Agent Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal placeholder reward for Navi OPD smoke tests.

Direct distillation with ``loss_mode=forward_kl_topk``,
``use_policy_gradient=False``, and ``use_task_rewards=False`` does not use task
rewards for the gradient. The v0 trainer still calls ``compute_score`` during
postprocessing, while the default dispatcher rejects unknown data sources.
Returning a constant score keeps that compatibility path operational without
changing the direct distillation objective.
"""

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 1.0
