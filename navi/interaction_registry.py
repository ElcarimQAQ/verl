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
"""Vendored interaction registry for the navi recipe.

Loads an interaction config YAML and instantiates each declared interaction
class. This replaces the upstream
``verl.interactions.utils.interaction_registry`` which was removed together
with the rest of the ``verl.interactions`` subsystem in this verl version.

Expected YAML structure (see ``navi/config/navi_interaction_config.yaml``)::

    interaction:
      - name: "navi"
        class_name: "navi.navi_interaction.NaviInteraction"
        config:
          user_model: "openai/Qwen3-235B-A22B-Thinking-2507-FP8"
          api_base: "http://localhost:12200/v1"
          ...
"""

import importlib
import logging
import os
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from navi.interaction_base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def get_interaction_class(class_name: str):
    """Resolve a dotted ``module.ClassName`` path to the class object."""
    module_name, cls_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def initialize_interactions_from_config(interaction_config_file: str) -> dict[str, "BaseInteraction"]:
    """Build ``{name: interaction_instance}`` from an interaction config YAML.

    Each item under the top-level ``interaction`` list must provide a unique
    ``name``, a ``class_name`` (dotted import path), and an optional ``config``
    mapping forwarded to the interaction constructor. ``name`` is injected into
    that ``config`` so subclasses can read it back.
    """
    interaction_config = OmegaConf.load(interaction_config_file)
    interaction_items = interaction_config.get("interaction", [])

    interaction_map: dict[str, "BaseInteraction"] = {}
    for item in interaction_items:
        name = item.get("name")
        if name is None:
            raise ValueError(f"Interaction config item missing required 'name': {item}")
        if name in interaction_map:
            raise ValueError(f"Duplicate interaction name '{name}' in {interaction_config_file}")

        class_name = item.get("class_name")
        if class_name is None:
            raise ValueError(f"Interaction '{name}' missing required 'class_name'")

        raw_config = item.get("config", {})
        config = OmegaConf.to_container(raw_config, resolve=True) if raw_config else {}
        config["name"] = name

        cls = get_interaction_class(class_name)
        interaction_map[name] = cls(config=config)
        logger.info(f"[interaction_registry] Loaded interaction '{name}' -> {class_name}")

    return interaction_map
