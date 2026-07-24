import copy
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from comfyui.cloud.client import CloudComfyUIClient
from config_loader import config
from utils.constants import BATCH_SIZE, WORKFLOW_TIMEOUT
from utils.execution_controller import get_execution_controller, should_continue_processing, stopped_prefix
from utils.images import save_uploaded_image
from utils.validation import (
    list_image_files,
    require_existing_dir,
    require_folder_name,
    require_non_empty,
)
from utils.workflow_runner import load_workflow_json

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "gpt_2imgbatch2img_api_cloud"

LOAD_IMAGE_1_NODE = "9"
LOAD_IMAGE_2_NODE = "10"
GPT_NODE = "6"
SAVE_IMAGE_NODE = "11"


@dataclass(frozen=True)
class GptPairTask:
    index: int
    total: int
    image1_path: Path
    image2_path: Path
    output_path: Path
    prompt: str
    label: str


def _process_gpt_pair(task: GptPairTask, api_schema_base: dict) -> tuple[bool, str]:
    client = CloudComfyUIClient()

    try:
        logger.info("GPT cloud %s: starting", task.label)

        image1_cloud = client.upload_image(task.image1_path)
        image2_cloud = client.upload_image(task.image2_path)

        api_schema = copy.deepcopy(api_schema_base)
        api_schema[LOAD_IMAGE_1_NODE]["inputs"]["image"] = image1_cloud
        api_schema[LOAD_IMAGE_2_NODE]["inputs"]["image"] = image2_cloud
        api_schema[GPT_NODE]["inputs"]["prompt"] = task.prompt

        prompt_id = client.submit_workflow(api_schema)
        logger.info(
            "GPT cloud %s: queued (Prompt ID=%s), waiting for completion...",
            task.label,
            prompt_id,
        )

        job = client.wait_for_completion(prompt_id, timeout=WORKFLOW_TIMEOUT)
        output_file = client.extract_primary_output_file(job, SAVE_IMAGE_NODE)

        image_bytes = client.download_output_file(output_file)
        if not image_bytes:
            raise RuntimeError("ComfyUI Cloud вернул пустой файл")

        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        with task.output_path.open("wb") as result_file:
            result_file.write(image_bytes)

        logger.info(
            "GPT cloud %s: saved %s (%d bytes)",
            task.label,
            task.output_path,
            len(image_bytes),
        )
        return True, f"[OK] {task.label}"

    except Exception as error:
        logger.exception("GPT cloud %s: failed", task.label)
        return False, f"[ОШИБКА] {task.label}: {error}"


def _run_gpt_tasks(tasks: list[GptPairTask], api_schema_base: dict) -> tuple[list[tuple[bool, str]], int]:
    results: list[tuple[bool, str]] = []
    completed_count = 0
    pending = list(tasks)
    active_futures: dict = {}

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        while pending or active_futures:
            while (
                pending
                and len(active_futures) < BATCH_SIZE
                and should_continue_processing()
            ):
                task = pending.pop(0)
                future = executor.submit(_process_gpt_pair, task, api_schema_base)
                active_futures[future] = task

            if not active_futures:
                break

            done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task = active_futures.pop(future)
                try:
                    success, message = future.result()
                except Exception as error:
                    success = False
                    message = f"[ОШИБКА] {task.label}: {error}"
                    logger.exception("Unexpected error in worker for %s", task.label)

                results.append((success, message))
                completed_count += 1
                logger.info(
                    "GPT cloud progress: %d/%d completed — %s",
                    completed_count,
                    len(tasks),
                    message,
                )

        if pending and get_execution_controller().is_stop_requested():
            logger.info(
                "Stop requested: %d GPT cloud tasks were not started",
                len(pending),
            )

    return results, len(tasks) - len(results)


def _format_batch_result(
    results: list[tuple[bool, str]],
    total: int,
    skipped: int,
    summary_lines: list[str],
) -> str:
    processed = sum(1 for success, _ in results if success)
    errors = sum(1 for success, _ in results if not success)
    details = "\n".join(message for _, message in results)
    prefix = stopped_prefix() if get_execution_controller().is_stop_requested() else ""

    summary = "\n".join(summary_lines)
    return (
        f"{prefix}"
        f"{summary}\n"
        f"Обработано успешно: {processed}/{total}\n"
        f"Ошибок: {errors}/{total}\n"
        f"Не запущено: {skipped}/{total}\n"
        f"Concurrency: {BATCH_SIZE}\n\n"
        f"Детали:\n{details}"
    )


def gpt_imgbatch2img_cloud(
    base_image,
    input_photos_path: str,
    output_result_folder: str,
    prompt: str,
):
    """
    Генерация по базовому изображению и папке с фотографиями через ComfyUI Cloud.
    Одновременно выполняется до 4 workflow; при завершении одного запускается следующий.
    """
    if base_image is None:
        raise gr.Error("Загрузите базовое изображение")

    prompt = require_non_empty(prompt, "Введите промпт")
    input_photos_path = require_non_empty(input_photos_path, "Введите путь к папке с фотографиями")
    output_result_folder = require_non_empty(output_result_folder, "Введите название папки для результатов")
    require_folder_name(output_result_folder)

    images_path = require_existing_dir(input_photos_path, "Указанный путь с изображениями не существует")
    image_files = list_image_files(images_path)

    if not image_files:
        raise gr.Error("В указанной папке нет фотографий")

    saved_image_path = save_uploaded_image(base_image, config.comfyui_portable_default_input_dir)
    if not saved_image_path:
        raise gr.Error("Не удалось сохранить входное изображение")

    base_image_path = Path(saved_image_path)
    result_images_path = Path(input_photos_path) / output_result_folder
    result_images_path.mkdir(parents=True, exist_ok=True)

    try:
        api_schema_base = load_workflow_json(WORKFLOW_NAME)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logger.exception("Workflow config error")
        raise gr.Error(f"Ошибка workflow:\n{error}") from error

    total = len(image_files)
    tasks = [
        GptPairTask(
            index=index,
            total=total,
            image1_path=base_image_path,
            image2_path=batch_file,
            output_path=result_images_path / batch_file.name,
            prompt=prompt,
            label=f"{index}/{total} ({batch_file.name})",
        )
        for index, batch_file in enumerate(image_files, start=1)
    ]

    logger.info(
        "GPT cloud imgbatch2img started: %d images, concurrency=%d",
        total,
        BATCH_SIZE,
    )

    results, skipped = _run_gpt_tasks(tasks, api_schema_base)

    return _format_batch_result(
        results,
        total,
        skipped,
        summary_lines=[f"Папка:\n{result_images_path}"],
    )


def gpt_batchbatch2img_cloud(
    input_photos_path_1: str,
    input_photos_path_2: str,
    prompt: str,
):
    """
    Каждое изображение из первой папки прогоняется через все изображения второй папки.
    Результаты складываются в подпапки по имени файла из первой папки.
    """
    prompt = require_non_empty(prompt, "Введите промпт")
    input_photos_path_1 = require_non_empty(input_photos_path_1, "Введите путь к первой папке")
    input_photos_path_2 = require_non_empty(input_photos_path_2, "Введите путь ко второй папке")

    input_path_1 = require_existing_dir(input_photos_path_1, "Первая папка не существует")
    input_path_2 = require_existing_dir(input_photos_path_2, "Вторая папка не существует")

    input_1_files = list_image_files(input_path_1)
    input_2_files = list_image_files(input_path_2)

    if not input_1_files:
        raise gr.Error("В первой папке нет изображений")

    if not input_2_files:
        raise gr.Error("Во второй папке нет изображений")

    try:
        api_schema_base = load_workflow_json(WORKFLOW_NAME)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logger.exception("Workflow config error")
        raise gr.Error(f"Ошибка workflow:\n{error}") from error

    tasks: list[GptPairTask] = []
    total = len(input_1_files) * len(input_2_files)
    task_index = 0

    for first_file in input_1_files:
        result_dir = input_path_1 / first_file.stem
        result_dir.mkdir(parents=True, exist_ok=True)

        for second_file in input_2_files:
            task_index += 1
            tasks.append(
                GptPairTask(
                    index=task_index,
                    total=total,
                    image1_path=first_file,
                    image2_path=second_file,
                    output_path=result_dir / second_file.name,
                    prompt=prompt,
                    label=(
                        f"{task_index}/{total} "
                        f"({first_file.name} + {second_file.name})"
                    ),
                )
            )

    logger.info(
        "GPT cloud batchbatch2img started: %d generations, concurrency=%d",
        total,
        BATCH_SIZE,
    )

    results, skipped = _run_gpt_tasks(tasks, api_schema_base)

    return _format_batch_result(
        results,
        total,
        skipped,
        summary_lines=[
            f"Моделей (папка 1): {len(input_1_files)}",
            f"Референсов (папка 2): {len(input_2_files)}",
            f"Всего генераций: {total}",
            f"Папка результатов: {input_path_1}",
        ],
    )
