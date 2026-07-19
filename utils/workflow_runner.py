import json
import logging
from typing import Any

from config_loader import config
from comfyui.portable.client import portable_client
from utils.constants import BATCH_SIZE, WORKFLOW_TIMEOUT

logger = logging.getLogger(__name__)


def load_workflow_json(workflow_name: str) -> dict[str, Any]:
    with open(config.get_workflow_path(workflow_name), "r", encoding="utf-8") as file_json:
        return json.load(file_json)


def total_batches(item_count: int, batch_size: int = BATCH_SIZE) -> int:
    return (item_count + batch_size - 1) // batch_size


def batch_start_index(batch_index: int, batch_size: int = BATCH_SIZE) -> int:
    return (batch_index - 1) * batch_size


def configure_loader_indices(
    api_schema: dict[str, Any],
    nodes: list[dict[str, str]],
    active_count: int,
    start_index: int,
) -> None:
    for i in range(active_count):
        loader = nodes[i]["loader"]
        api_schema[loader]["inputs"]["index"] = start_index + i


def remove_unused_branches(
    api_schema: dict[str, Any],
    nodes: list[dict[str, str]],
    active_count: int,
) -> None:
    for i in range(active_count, BATCH_SIZE):
        for node_id in nodes[i].values():
            api_schema.pop(node_id, None)


def queue_and_wait(
    api_schema: dict[str, Any],
    batch_index: int,
    total: int,
    *,
    timeout: int = WORKFLOW_TIMEOUT,
    queued_log: str,
) -> str:
    prompt_id = portable_client.submit_workflow(api_schema)
    logger.info(queued_log, batch_index, total, prompt_id)
    portable_client.wait_for_completion(prompt_id, timeout=timeout)
    return prompt_id
