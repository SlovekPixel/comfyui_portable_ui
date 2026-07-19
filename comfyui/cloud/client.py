import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.exceptions import HTTPError, RequestException

from config_loader import config
from comfyui.errors import extract_execution_error, format_execution_error
from utils.constants import (
    HTTP_CONNECTION_CHECK_TIMEOUT,
    HTTP_REQUEST_TIMEOUT,
    POLL_INTERVAL,
)

logger = logging.getLogger(__name__)

OUTPUT_MEDIA_KEYS = ("images", "video", "audio", "a_images", "b_images")

# /api/job/{id}/status возвращает "success", /api/jobs/{id} — "completed"
COMPLETED_JOB_STATUSES = frozenset({"completed", "success"})
FAILED_JOB_STATUSES = frozenset({"failed", "cancelled", "error"})


class CloudComfyUIClient:
    """Клиент для ComfyUI Cloud API."""

    def __init__(self) -> None:
        self._session = requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": config.comfyui_cloud_api_key}

    def _url(self, path: str) -> str:
        return f"{config.comfyui_cloud_base_url.rstrip('/')}{path}"

    @staticmethod
    def _raise_for_cloud_response(response: requests.Response, action: str) -> None:
        try:
            response.raise_for_status()
        except HTTPError as error:
            if response.status_code == 401:
                raise RuntimeError(
                    "ComfyUI Cloud: неверный API-ключ. "
                    "Проверьте COMFYUI_CLOUD_API_KEY в .env "
                    "(ключ создаётся на platform.comfy.org)."
                ) from error
            raise RuntimeError(
                f"ComfyUI Cloud: ошибка {action} (HTTP {response.status_code})."
            ) from error

    def upload_image(self, file_path: Path) -> str:
        logger.info("Cloud upload: %s", file_path)
        with file_path.open("rb") as image_file:
            response = self._session.post(
                self._url("/api/upload/image"),
                headers=self._headers,
                files={"image": (file_path.name, image_file)},
                data={"type": "input", "overwrite": "true"},
                timeout=HTTP_REQUEST_TIMEOUT,
            )
        self._raise_for_cloud_response(response, "загрузки изображения")
        result = response.json()
        cloud_name = result["name"]
        logger.info("Cloud upload done: local=%s -> cloud=%s", file_path.name, cloud_name)
        return cloud_name

    def submit_workflow(self, workflow: dict) -> str:
        logger.info("Cloud submit workflow (%d nodes)", len(workflow))
        payload = {
            "prompt": workflow,
            "extra_data": {"api_key_comfy_org": config.comfyui_cloud_api_key},
        }
        response = self._session.post(
            self._url("/api/prompt"),
            headers={**self._headers, "Content-Type": "application/json"},
            json=payload,
            timeout=HTTP_REQUEST_TIMEOUT,
        )
        self._raise_for_cloud_response(response, "отправки workflow")
        result = response.json()

        if "error" in result and result["error"]:
            raise RuntimeError(f"ComfyUI Cloud workflow error: {result['error']}")

        prompt_id = result["prompt_id"]
        logger.info("Cloud workflow queued: prompt_id=%s", prompt_id)
        return prompt_id

    def get_job(self, prompt_id: str) -> dict[str, Any]:
        logger.info("Cloud get job: %s", prompt_id)
        response = self._session.get(
            self._url(f"/api/jobs/{prompt_id}"),
            headers=self._headers,
            timeout=HTTP_REQUEST_TIMEOUT,
        )
        self._raise_for_cloud_response(response, "получения job")
        job = response.json()
        logger.info(
            "Cloud job received: id=%s status=%s outputs_count=%s",
            job.get("id"),
            job.get("status"),
            job.get("outputs_count"),
        )
        return job

    def get_job_status(self, prompt_id: str) -> str:
        response = self._session.get(
            self._url(f"/api/job/{prompt_id}/status"),
            headers=self._headers,
            timeout=HTTP_REQUEST_TIMEOUT,
        )
        self._raise_for_cloud_response(response, "получения статуса job")
        payload = response.json()
        status = payload.get("status", "")
        logger.info(
            "Cloud job status poll: prompt_id=%s status=%s",
            prompt_id,
            status,
        )
        if payload.get("error_message"):
            logger.warning(
                "Cloud job status error_message: %s",
                payload["error_message"],
            )
        return status

    @staticmethod
    def _is_job_completed(status: str) -> bool:
        return status in COMPLETED_JOB_STATUSES

    @staticmethod
    def _is_job_failed(status: str) -> bool:
        return status in FAILED_JOB_STATUSES

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 300,
        poll_interval: float = POLL_INTERVAL,
    ) -> dict[str, Any]:
        logger.info(
            "Cloud wait for completion: prompt_id=%s timeout=%ss poll=%ss",
            prompt_id,
            timeout,
            poll_interval,
        )
        deadline = time.time() + timeout
        poll_number = 0

        while time.time() < deadline:
            poll_number += 1
            status = self.get_job_status(prompt_id)
            elapsed = int(time.time() - (deadline - timeout))

            if self._is_job_completed(status):
                logger.info(
                    "Cloud job finished: prompt_id=%s status=%s polls=%d elapsed=%ss",
                    prompt_id,
                    status,
                    poll_number,
                    elapsed,
                )
                return self.get_job(prompt_id)

            if self._is_job_failed(status):
                raise RuntimeError(f"ComfyUI Cloud job завершился со статусом: {status}")

            logger.info(
                "Cloud job in progress: prompt_id=%s status=%s poll=%d elapsed=%ss",
                prompt_id,
                status,
                poll_number,
                elapsed,
            )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"ComfyUI Cloud job {prompt_id} не завершился за {timeout} секунд "
            f"(последний статус: {status})."
        )

    def get_job_outputs(self, prompt_id: str) -> dict[str, Any]:
        job = self.get_job(prompt_id)
        execution_status = job.get("execution_status", {})
        status = job.get("status", execution_status)

        if isinstance(status, dict):
            status_str = status.get("status_str", "")
            if status_str == "error":
                raise RuntimeError(
                    format_execution_error(status, extract_execution_error(job))
                )

        outputs = job.get("outputs")
        if outputs:
            return outputs

        raise RuntimeError(f"ComfyUI Cloud: job {prompt_id} не содержит outputs.")

    @staticmethod
    def extract_primary_output_file(
        job: dict[str, Any],
        save_node_id: str,
    ) -> dict[str, Any]:
        outputs = job.get("outputs", {})
        node_outputs = outputs.get(save_node_id, {})

        for key in OUTPUT_MEDIA_KEYS:
            files = node_outputs.get(key)
            if files:
                output_file = files[0]
                logger.info(
                    "Cloud output file from node %s/%s: display_name=%s filename=%s",
                    save_node_id,
                    key,
                    output_file.get("display_name"),
                    output_file.get("filename"),
                )
                return output_file

        preview_output = job.get("preview_output")
        if preview_output and preview_output.get("filename"):
            logger.info(
                "Cloud output file from preview_output: filename=%s",
                preview_output.get("filename"),
            )
            return preview_output

        raise RuntimeError(
            f"ComfyUI Cloud не вернул файл результата для node {save_node_id}."
        )

    def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output",
    ) -> bytes:
        logger.info(
            "Cloud download: filename=%s subfolder=%s type=%s",
            filename,
            subfolder,
            output_type,
        )
        response = self._session.get(
            self._url("/api/view"),
            headers=self._headers,
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": output_type,
            },
            timeout=HTTP_REQUEST_TIMEOUT,
            allow_redirects=False,
        )

        if response.status_code in (301, 302, 303, 307, 308):
            signed_url = response.headers.get("Location")
            if not signed_url:
                raise RuntimeError("ComfyUI Cloud: /api/view вернул redirect без Location.")

            logger.info(
                "Cloud download redirect: HTTP %s -> %s",
                response.status_code,
                signed_url[:120],
            )
            file_response = self._session.get(
                signed_url,
                timeout=HTTP_REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            self._raise_for_cloud_response(file_response, "скачивания файла")
            logger.info("Cloud download done: %d bytes", len(file_response.content))
            return file_response.content

        self._raise_for_cloud_response(response, "скачивания файла")
        logger.info("Cloud download done (direct): %d bytes", len(response.content))
        return response.content

    def download_output_file(self, output_file: dict[str, Any]) -> bytes:
        return self.download_output(
            filename=output_file["filename"],
            subfolder=output_file.get("subfolder", ""),
            output_type=output_file.get("type", "output"),
        )

    def check_connection(self) -> bool:
        try:
            response = self._session.get(
                self._url("/api/system_stats"),
                timeout=HTTP_CONNECTION_CHECK_TIMEOUT,
            )
            return response.status_code == 200
        except RequestException:
            return False


cloud_client = CloudComfyUIClient()
