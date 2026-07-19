import logging
import time

import requests
from requests.exceptions import RequestException

from config_loader import config
from comfyui.errors import extract_execution_error, format_execution_error
from utils.constants import (
    HTTP_CONNECTION_CHECK_TIMEOUT,
    HTTP_REQUEST_TIMEOUT,
    POLL_INTERVAL,
)

logger = logging.getLogger(__name__)


class PortableComfyUIClient:
    """Клиент для локальной Portable-версии ComfyUI."""

    def __init__(self) -> None:
        self._session = requests.Session()

    def submit_workflow(self, workflow: dict) -> str:
        payload = {
            "extra_data": {"api_key_comfy_org": config.comfyui_cloud_api_key},
            "prompt": workflow,
        }
        response = self._session.post(
            config.comfyui_portable_prompt_url,
            json=payload,
            timeout=HTTP_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["prompt_id"]

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 300,
        poll_interval: float = POLL_INTERVAL,
    ) -> dict:
        deadline = time.time() + timeout

        while time.time() < deadline:
            response = self._session.get(
                f"{config.comfyui_portable_history_url}/{prompt_id}",
                timeout=HTTP_REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            history = response.json()

            if prompt_id not in history:
                time.sleep(poll_interval)
                continue

            job = history[prompt_id]
            status = job.get("status", {})
            status_str = status.get("status_str", "")

            if status_str == "error":
                raise RuntimeError(
                    format_execution_error(status, extract_execution_error(job))
                )

            if status.get("completed", False):
                return job.get("outputs", {})

            time.sleep(poll_interval)

        raise TimeoutError(f"Workflow не завершился за {timeout} секунд.")

    def check_connection(self) -> bool:
        try:
            response = self._session.get(
                config.comfyui_portable_prompt_url,
                timeout=HTTP_CONNECTION_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except RequestException:
            return False


portable_client = PortableComfyUIClient()
