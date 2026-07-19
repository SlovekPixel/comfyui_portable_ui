import copy
import json
import logging
from pathlib import Path

import gradio as gr
from requests.exceptions import RequestException

from config_loader import config
from utils.constants import BATCH_SIZE
from utils.prompts import chunks
from utils.validation import list_image_files, require_existing_dir, require_non_empty
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

WORKFLOW_NAME = "nbp_batchbatch2img_4x_api"

NODES = [
    {"loader": "22", "batch": "82", "gemini": "83", "save": "24"},
    {"loader": "27", "batch": "84", "gemini": "85", "save": "28"},
    {"loader": "30", "batch": "86", "gemini": "87", "save": "31"},
    {"loader": "33", "batch": "88", "gemini": "89", "save": "34"},
]


def nbp_batchbatch2img_4x_api(
    input_photos_path_1,
    input_photos_path_2,
    prompt,
):
    """
    Каждая фотография из input_jeans_models
    прогоняется через все фотографии input_voronka
    по 4 изображения одновременно.
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

        batches_total_2 = total_batches(len(input_2_files))
        processed_generations = 0

        stop_requested = False

        for first_index, first_file in enumerate(input_1_files):
            if not should_continue_processing():
                logger.info(
                    "Stop requested before model %d/%d, skipping remaining work",
                    first_index + 1,
                    len(input_1_files),
                )
                stop_requested = True
                break

            logger.info(
                "Processing first folder files %d/%d (%s)",
                first_index + 1,
                len(input_1_files),
                first_file.name,
            )

            for batch_index, batch in enumerate(chunks(input_2_files, BATCH_SIZE), start=1):
                if not should_continue_processing():
                    logger.info(
                        "Stop requested before batch %d/%d, skipping remaining batches",
                        batch_index,
                        batches_total_2,
                    )
                    stop_requested = True
                    break

                logger.info(
                    "Processing batch %d/%d (%d reference images)",
                    batch_index,
                    batches_total_2,
                    len(batch),
                )

                api_schema = copy.deepcopy(api_schema_base)

                api_schema["81"]["inputs"]["value"] = str(input_path_1)
                api_schema["23"]["inputs"]["value"] = str(input_path_2)
                api_schema["25"]["inputs"]["value"] = first_file.stem
                api_schema["47"]["inputs"]["value"] = prompt
                api_schema["80"]["inputs"]["index"] = first_index

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
                    batches_total_2,
                    queued_log="Queued batch %d/%d (prompt_id=%s)",
                )

                processed_generations += len(batch)

                logger.info("Completed batch %d/%d", batch_index, batches_total_2)

            if stop_requested:
                break

        logger.info(
            "Finished.\n"
            "Models: %d\n"
            "Reference images: %d\n"
            "Generations: %d",
            len(input_2_files),
            len(input_1_files),
            processed_generations,
        )

        prefix = stopped_prefix() if get_execution_controller().is_stop_requested() else ""
        return (
            f"{prefix}"
            f"Обработано моделей: {len(input_2_files)}\n"
            f"Обработано референсов: {len(input_1_files)}\n"
            f"Всего генераций: {processed_generations}"
        )

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Workflow config error: %s", e)
        raise gr.Error(f"Ошибка конфига воркфлоу: {e}")

    except RequestException as e:
        logger.error("ComfyUI connection error: %s", e)
        raise gr.Error(f"Нет соединения с ComfyUI Portable ({config.comfyui_portable_prompt_url})")

    except (RuntimeError, TimeoutError) as e:
        logger.error("Job error: %s", e)
        raise gr.Error(str(e))

    except Exception as e:
        logger.exception("Unexpected error")
        raise gr.Error(f"Неожиданная ошибка: {e}")
