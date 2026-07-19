import copy
import json
import logging
import os
from pathlib import Path

import gradio as gr
from requests.exceptions import RequestException

from config_loader import config
from utils.constants import BATCH_SIZE
from utils.images import save_uploaded_image
from utils.prompts import chunks
from utils.validation import (
    list_image_files,
    require_existing_dir,
    require_folder_name,
    require_non_empty,
)
from utils.execution_controller import get_execution_controller, should_continue_processing, stopped_prefix
from utils.workflow_runner import (
    batch_start_index,
    configure_loader_indices,
    load_workflow_json,
    queue_and_wait,
    remove_unused_branches,
    total_batches,
)

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "nbp_imgbatch2img_4x_api"

NODES = [
    {"loader": "22", "batch": "86", "gemini": "87", "save": "24"},
    {"loader": "27", "batch": "84", "gemini": "85", "save": "28"},
    {"loader": "30", "batch": "82", "gemini": "83", "save": "31"},
    {"loader": "33", "batch": "80", "gemini": "81", "save": "34"},
]


def nbp_imgbatch2img_4x_api(
    base_image,
    input_photos_path,
    output_result_folder,
    prompt,
):
    """
    Генерация изображений по одной базовой фотографии и папке со входными фото.
    За один запуск ComfyUI обрабатывается до 4 фотографий одновременно.
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

    try:
        api_schema_base = load_workflow_json(WORKFLOW_NAME)

        saved_image_path = save_uploaded_image(base_image, config.comfyui_portable_default_input_dir)
        if not saved_image_path:
            raise ValueError("Не удалось сохранить входное изображение")

        image_filename = os.path.basename(saved_image_path)

        result_images_path = os.path.join(input_photos_path, output_result_folder)
        os.makedirs(result_images_path, exist_ok=True)

        batches_total = total_batches(len(image_files))
        processed = 0

        for batch_index, batch in enumerate(chunks(image_files, BATCH_SIZE), start=1):
            if not should_continue_processing():
                logger.info(
                    "Stop requested before batch %d/%d, skipping remaining batches",
                    batch_index,
                    batches_total,
                )
                break

            logger.info(
                "Processing batch %d/%d (%d images)",
                batch_index,
                batches_total,
                len(batch),
            )

            api_schema = copy.deepcopy(api_schema_base)

            api_schema["23"]["inputs"]["value"] = str(images_path)
            api_schema["25"]["inputs"]["value"] = output_result_folder
            api_schema["47"]["inputs"]["value"] = prompt
            api_schema["51"]["inputs"]["image"] = image_filename

            configure_loader_indices(
                api_schema,
                NODES,
                len(batch),
                batch_start_index(batch_index),
            )
            remove_unused_branches(api_schema, NODES, len(batch))

            queue_and_wait(
                api_schema,
                batch_index,
                batches_total,
                queued_log="Queued batch %d/%d (Prompt ID=%s)",
            )

            processed += len(batch)

            logger.info("Completed batch %d/%d", batch_index, batches_total)

        logger.info(
            "Completed. %d images processed. Results: %s",
            processed,
            result_images_path,
        )

        prefix = stopped_prefix() if get_execution_controller().is_stop_requested() else ""
        return (
            f"{prefix}"
            f"Количество обработанных фотографий: {processed}\n"
            f"Папка:\n{result_images_path}"
        )

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.exception("Workflow config error")
        raise gr.Error(f"Ошибка workflow:\n{e}")

    except RequestException:
        logger.exception("ComfyUI connection error")
        raise gr.Error(f"Нет соединения с ComfyUI Portable ({config.comfyui_portable_prompt_url})")

    except (RuntimeError, TimeoutError) as e:
        logger.exception("Workflow execution error")
        raise gr.Error(str(e))

    except Exception as e:
        logger.exception("Unexpected error")
        raise gr.Error(f"Неожиданная ошибка:\n{e}")
