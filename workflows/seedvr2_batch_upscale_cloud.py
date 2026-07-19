import copy
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import gradio as gr

from comfyui.cloud.client import CloudComfyUIClient
from utils.constants import WORKFLOW_TIMEOUT
from utils.execution_controller import get_execution_controller, should_continue_processing, stopped_prefix
from utils.validation import (
    list_image_files,
    require_existing_dir,
    require_folder_name,
    require_non_empty,
)
from utils.workflow_runner import load_workflow_json

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "seedvr2-upscaler-fast-api-cloud"

LOAD_IMAGE_NODE = "80"
WIDTH_SCALE_NODE = "83"
SAVE_IMAGE_NODE = "91"

MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 5


def _process_single_image(
    image_file: Path,
    index: int,
    total: int,
    api_schema_base: dict,
    width_scale: int,
    result_images_path: str,
) -> tuple[bool, str]:
    """
    Обрабатывает одно изображение через ComfyUI Cloud API.
    Возвращает (успех, сообщение). Ошибки не пробрасываются наружу.
    """
    client = CloudComfyUIClient()
    label = f"{index}/{total} ({image_file.name})"

    try:
        logger.info("Cloud upscale %s: starting", label)

        uploaded_name = client.upload_image(image_file)
        logger.info("Cloud upscale %s: uploaded as %s", label, uploaded_name)

        api_schema = copy.deepcopy(api_schema_base)
        api_schema[LOAD_IMAGE_NODE]["inputs"]["image"] = uploaded_name
        api_schema[WIDTH_SCALE_NODE]["inputs"]["value"] = int(width_scale)

        prompt_id = client.submit_workflow(api_schema)
        logger.info(
            "Cloud upscale %s: queued (Prompt ID=%s), waiting for completion...",
            label,
            prompt_id,
        )

        job = client.wait_for_completion(prompt_id, timeout=WORKFLOW_TIMEOUT)
        output_file = client.extract_primary_output_file(job, SAVE_IMAGE_NODE)

        logger.info(
            "Cloud upscale %s: downloading result (cloud file: %s)",
            label,
            output_file.get("display_name") or output_file["filename"],
        )

        image_bytes = client.download_output_file(output_file)
        if not image_bytes:
            raise RuntimeError("ComfyUI Cloud вернул пустой файл")

        destination_path = os.path.join(result_images_path, image_file.name)
        with open(destination_path, "wb") as result_file:
            result_file.write(image_bytes)

        logger.info(
            "Cloud upscale %s: saved %s (%d bytes)",
            label,
            destination_path,
            len(image_bytes),
        )
        return True, f"[OK] {image_file.name}"

    except Exception as error:
        logger.exception("Cloud upscale %s: failed", label)
        return False, f"[ОШИБКА] {image_file.name}: {error}"


def seedvr2_batch_upscale_cloud(
    input_photos_path: str,
    output_result_folder: str,
    width_scale: int,
    concurrency: int = 1,
):
    """
    Пакетный upscale изображений через ComfyUI Cloud API.
    Одновременно выполняется не более `concurrency` workflow (1–5).
    """
    input_photos_path = require_non_empty(
        input_photos_path,
        "Введите путь к папке с фотографиями",
    )
    output_result_folder = require_non_empty(
        output_result_folder,
        "Введите название папки для результатов",
    )
    require_folder_name(output_result_folder)

    if width_scale is None or width_scale <= 0:
        raise gr.Error("Width Scale должен быть положительным числом")

    if concurrency is None:
        concurrency = MIN_CONCURRENCY
    concurrency = int(concurrency)
    if concurrency < MIN_CONCURRENCY or concurrency > MAX_CONCURRENCY:
        raise gr.Error(
            f"Concurrency должен быть от {MIN_CONCURRENCY} до {MAX_CONCURRENCY}"
        )

    images_path = require_existing_dir(
        input_photos_path,
        "Указанный путь с изображениями не существует",
    )
    image_files = list_image_files(images_path)

    if not image_files:
        raise gr.Error("В указанной папке нет фотографий")

    result_images_path = os.path.join(input_photos_path, output_result_folder)
    os.makedirs(result_images_path, exist_ok=True)

    try:
        api_schema_base = load_workflow_json(WORKFLOW_NAME)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.exception("Workflow config error")
        raise gr.Error(f"Ошибка workflow:\n{e}")

    total = len(image_files)
    logger.info(
        "Cloud upscale batch started: %d images, concurrency=%d",
        total,
        concurrency,
    )

    results: list[tuple[bool, str]] = []
    completed_count = 0
    pending = list(enumerate(image_files, start=1))
    active_futures: dict = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        while pending or active_futures:
            while (
                pending
                and len(active_futures) < concurrency
                and should_continue_processing()
            ):
                index, image_file = pending.pop(0)
                future = executor.submit(
                    _process_single_image,
                    image_file,
                    index,
                    total,
                    api_schema_base,
                    int(width_scale),
                    result_images_path,
                )
                active_futures[future] = image_file

            if not active_futures:
                break

            done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                image_file = active_futures.pop(future)
                try:
                    success, message = future.result()
                except Exception as error:
                    success = False
                    message = f"[ОШИБКА] {image_file.name}: {error}"
                    logger.exception("Unexpected error in worker for %s", image_file.name)

                results.append((success, message))
                completed_count += 1
                logger.info(
                    "Cloud upscale progress: %d/%d completed — %s",
                    completed_count,
                    total,
                    message,
                )

        if pending and get_execution_controller().is_stop_requested():
            logger.info(
                "Stop requested: %d cloud tasks were not started",
                len(pending),
            )

    processed = sum(1 for success, _ in results if success)
    errors = sum(1 for success, _ in results if not success)
    skipped = total - len(results)

    logger.info(
        "Cloud upscale completed. Success: %d, errors: %d, skipped: %d. Results: %s",
        processed,
        errors,
        skipped,
        result_images_path,
    )

    details = "\n".join(message for _, message in results)
    prefix = stopped_prefix() if get_execution_controller().is_stop_requested() else ""
    return (
        f"{prefix}"
        f"Обработано успешно: {processed}/{total}\n"
        f"Ошибок: {errors}/{total}\n"
        f"Не запущено: {skipped}/{total}\n"
        f"Concurrency: {concurrency}\n"
        f"Папка:\n{result_images_path}\n\n"
        f"Детали:\n{details}"
    )
