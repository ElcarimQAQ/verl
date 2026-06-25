# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Vendored ``BaseInteraction`` for the navi recipe.

The upstream ``verl.interactions`` subsystem was removed in this verl
version (``verl/workers/config/rollout.py`` no longer exposes
``interaction_config_path`` and ``AgentLoopWorker`` no longer builds an
``interaction_map``). The navi recipe drives its own user-simulator loop,
so it carries a minimal copy of the original ``BaseInteraction`` contract
here instead of importing ``verl.interactions.base``.
"""

from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    """Base class for an interaction (user simulator) used by the agent loop.

    Subclasses implement the lifecycle methods below. Instances are created
    once per agent-loop worker by ``initialize_interactions_from_config`` and
    reused across trajectories; per-trajectory state must be keyed by
    ``instance_id`` (see :class:`navi.navi_interaction.NaviInteraction`).
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Begin a new interaction instance and return its ``instance_id``."""
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Produce the simulated user reply for the current turn.

        Returns ``(should_terminate_sequence, response, reward, metrics)``.
        """
        raise NotImplementedError

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Optional turn-level score for this interaction instance."""
        raise NotImplementedError

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Release any state associated with ``instance_id``."""
        raise NotImplementedError
