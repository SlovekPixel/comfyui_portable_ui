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
from utils.prompts import chunks, split_prompts_by
from utils.validation import require_non_empty
from utils.execution_controller import get_execution_controller, should_continue_processing, stopped_prefix
from utils.workflow_runner import load_workflow_json, queue_and_wait, total_batches

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "nbp_imgprompts2img_4x_api"

NODES = [
    ("787", "790"),
    ("792", "791"),
    ("796", "795"),
    ("794", "793"),
]

GEMINI_NODES = [gemini_node for gemini_node, _ in NODES]
BATCH_IMAGES_NODE = "782"
SECOND_IMAGE_LOADER = "768"


def nbp_imgprompts2img_4x_api(
    base_image_1,
    base_image_2,
    prompt,
    output_photos_path,
):
    if base_image_1 is None:
        raise gr.Error("Загрузите первое изображение")

    prompt = require_non_empty(prompt, "Введите промпты, разделенные пустой строкой")
    output_photos_path = require_non_empty(output_photos_path, "Укажите путь к папке для результатов")

    output_dir = Path(output_photos_path)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise gr.Error(f"Не удалось создать папку для результатов:\n{e}")

    prompts = split_prompts_by(prompt, r"\n\s*\n+")

    if not prompts:
        raise gr.Error("Не найдено ни одного промпта")

    try:
        api_schema_base = load_workflow_json(WORKFLOW_NAME)

        saved_image1 = save_uploaded_image(base_image_1, config.comfyui_portable_default_input_dir)
        if not saved_image1:
            raise ValueError("Не удалось сохранить первое входное изображения")

        image1_name = os.path.basename(saved_image1)

        image2_name = None
        has_second_image = base_image_2 is not None
        if has_second_image:
            saved_image2 = save_uploaded_image(base_image_2, config.comfyui_portable_default_input_dir)
            if not saved_image2:
                raise ValueError("Не удалось сохранить второе входное изображение")

            image2_name = os.path.basename(saved_image2)

        batches_total = total_batches(len(prompts))
        processed = 0

        for batch_index, batch in enumerate(chunks(prompts, BATCH_SIZE), start=1):
            if not should_continue_processing():
                logger.info(
                    "Stop requested before batch %d/%d, skipping remaining batches",
                    batch_index,
                    batches_total,
                )
                break

            logger.info(
                "Processing batch %d/%d (%d prompts)",
                batch_index,
                batches_total,
                len(batch),
            )

            api_schema = copy.deepcopy(api_schema_base)

            api_schema["784"]["inputs"]["value"] = str(output_dir)
            api_schema["766"]["inputs"]["image"] = image1_name

            if has_second_image:
                api_schema[SECOND_IMAGE_LOADER]["inputs"]["image"] = image2_name
            else:
                del api_schema[SECOND_IMAGE_LOADER]
                del api_schema[BATCH_IMAGES_NODE]
                for gemini_node in GEMINI_NODES:
                    api_schema[gemini_node]["inputs"]["images"] = ["766", 0]

            for i, text_prompt in enumerate(batch):
                gemini_node = NODES[i][0]
                api_schema[gemini_node]["inputs"]["prompt"] = text_prompt

            for i in range(len(batch), BATCH_SIZE):
                gemini_node, save_node = NODES[i]

                if gemini_node in api_schema:
                    del api_schema[gemini_node]

                if save_node in api_schema:
                    del api_schema[save_node]

            queue_and_wait(
                api_schema,
                batch_index,
                batches_total,
                queued_log="Queued batch %d/%d. Prompt ID=%s",
            )

            processed += len(batch)

            logger.info("Finished batch %d/%d", batch_index, batches_total)

        prefix = stopped_prefix() if get_execution_controller().is_stop_requested() else ""
        return (
            f"{prefix}"
            f"Количество обработанных промптов: {processed}\n"
            f"Папка:\n{output_dir}"
        )

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.exception("Workflow error")
        raise gr.Error(f"Ошибка workflow:\n{e}")

    except RequestException as e:
        logger.exception("ComfyUI connection error")
        raise gr.Error(f"Нет соединения с ComfyUI\n\n{e}")

    except TimeoutError as e:
        logger.exception("Timeout")
        raise gr.Error(str(e))

    except RuntimeError as e:
        logger.exception("Runtime")
        raise gr.Error(str(e))

    except Exception as e:
        logger.exception("Unexpected error")
        raise gr.Error(f"Неожиданная ошибка:\n{e}")
