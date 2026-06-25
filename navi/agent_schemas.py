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
"""navi-specific ``AgentState`` and ``AgentData``.

The navi recipe runs a user-simulator turn that the upstream
``ToolAgentLoop`` state machine does not model:

- The upstream ``AgentState`` (``verl/experimental/agent_loop/tool_agent_loop.py``)
  only has ``PENDING / GENERATING / PROCESSING_TOOLS / TERMINATED`` — no
  ``INTERACTING``. A Python ``Enum`` with members cannot be extended, so the
  full set (including ``INTERACTING``) is declared fresh here.
- The upstream ``AgentData.__init__`` has a fixed signature with no
  ``interaction`` / ``interaction_kwargs``. We subclass it to carry the active
  interaction (user simulator) and its kwargs, and to make the multimodal
  arguments optional (navi is text-only).

Only ``navi/navi_agent_loop.py`` consumes these; ``NaviAgentLoop`` overrides
``run()`` and all state handlers except ``_handle_pending_state`` (inherited),
whose return value it discards — so the fresh ``AgentState`` enum stays
internal to navi's own loop.
"""

from enum import Enum
from typing import Any, Optional

from verl.experimental.agent_loop.tool_agent_loop import AgentData as _BaseAgentData


class AgentState(Enum):
    """Agent-loop states for navi, extending the upstream set with INTERACTING.

    Values match the upstream enum so existing comparisons/logging are
    unchanged; ``INTERACTING`` is the navi-specific user-simulator turn entered
    after a ``notify_user_msg`` tool call.
    """

    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData(_BaseAgentData):
    """Agent loop state container with user-simulator interaction support.

    Adds ``interaction`` / ``interaction_kwargs`` on top of the upstream
    ``AgentData`` and accepts text-only construction (multimodal args default
    to ``None``). All base attributes (``prompt_ids``, ``response_ids``,
    ``turn_scores``, ``tool_calls``, ``extra_fields``, ...) are inherited, so
    inherited ``ToolAgentLoop`` methods (e.g. ``_handle_pending_state``) keep
    working unchanged.
    """

    def __init__(
        self,
        *,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        interaction: Optional[Any] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            messages=messages,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        # navi-specific: active user simulator and its per-trajectory kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
