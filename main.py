import logging
import os
import sys

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
if _WEB_DIR not in sys.path:
    sys.path.insert(0, _WEB_DIR)

# VPN/системный proxy часто перехватывает localhost и ломает self-check Gradio при launch().
_LOCAL_NO_PROXY = "localhost,127.0.0.1,0.0.0.0"
for _proxy_var in ("NO_PROXY", "no_proxy"):
    _current = os.environ.get(_proxy_var, "")
    if _LOCAL_NO_PROXY not in _current:
        os.environ[_proxy_var] = ",".join(part for part in (_current, _LOCAL_NO_PROXY) if part)

import gradio as gr

from comfyui.cloud.client import cloud_client
from comfyui.portable.client import portable_client
from config_loader import config
from workflows.nbp_batchbatch2img_4x_api import nbp_batchbatch2img_4x_api
from workflows.nbp_imgbatch2img_4x_api import nbp_imgbatch2img_4x_api
from workflows.nbp_imgprompts2img_4x_api import nbp_imgprompts2img_4x_api
from utils.execution_controller import request_batch_stop, run_with_execution_control
from workflows.seedvr2_batch_upscale_cloud import seedvr2_batch_upscale_cloud

logger = logging.getLogger(__name__)


def _wire_stop(stop_btn: gr.Button, stop_status: gr.Textbox) -> None:
    stop_btn.click(fn=request_batch_stop, outputs=stop_status)


def _wire_generation(
    run_btn: gr.Button,
    handler,
    inputs,
    outputs,
) -> None:
    run_btn.click(
        fn=run_with_execution_control(handler),
        inputs=inputs,
        outputs=outputs,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=config.ui_title) as demo:
        with gr.Tab("Генерация обложек (NBP) 4x"):
            with gr.Row():
                with gr.Column():
                    base_image = gr.Image(type="pil", label="Исходное изображение")
                    input_photos_path = gr.Textbox(label="Путь к папке с фотографиями (C:\\Users\\Public...)")
                    output_result_folder = gr.Textbox(label="Название папки для результатов")
                    prompt = gr.TextArea(label="Текст промпта")
                    with gr.Row():
                        run_btn = gr.Button("Сгенерировать", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    stop_status = gr.Textbox(label="Статус остановки", interactive=False, lines=1)
                with gr.Column():
                    output = gr.Textbox(label="Результат генерации тут", interactive=False)

            _wire_generation(
                run_btn,
                nbp_imgbatch2img_4x_api,
                inputs=[base_image, input_photos_path, output_result_folder, prompt],
                outputs=output,
            )
            _wire_stop(stop_btn, stop_status)

        with gr.Tab("Генерация адаптива воронки (NBP) 4x"):
            with gr.Row():
                with gr.Column():
                    input_photos_path_1 = gr.Textbox(
                        label="Путь к папке 1 (На основе имён изображений в этой папке будут создаваться папки с результатами)",
                    )
                    input_photos_path_2 = gr.Textbox(label="Путь к папке 2")
                    prompt = gr.TextArea(label="Текст промпта")
                    with gr.Row():
                        run_btn = gr.Button("Сгенерировать", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    stop_status = gr.Textbox(label="Статус остановки", interactive=False, lines=1)
                with gr.Column():
                    output = gr.Textbox(label="Результат генерации тут", interactive=False)

            _wire_generation(
                run_btn,
                nbp_batchbatch2img_4x_api,
                inputs=[input_photos_path_1, input_photos_path_2, prompt],
                outputs=output,
            )
            _wire_stop(stop_btn, stop_status)

        with gr.Tab("Генерация воронки (NBP) 4x"):
            with gr.Row():
                with gr.Column():
                    base_image_1 = gr.Image(type="pil", label="Исходное изображение 1")
                    base_image_2 = gr.Image(type="pil", label="Исходное изображение 2 (не обязательно)")
                    prompt = gr.TextArea(label="Текст нескольких промптов (разделитель пустая строка)")
                    output_photos_path = gr.Textbox(label="Путь к папке для результатов (C:\\Users\\Public...)")
                    with gr.Row():
                        run_btn = gr.Button("Сгенерировать", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    stop_status = gr.Textbox(label="Статус остановки", interactive=False, lines=1)
                with gr.Column():
                    output = gr.Textbox(label="Результат генерации тут", interactive=False)

            _wire_generation(
                run_btn,
                nbp_imgprompts2img_4x_api,
                inputs=[base_image_1, base_image_2, prompt, output_photos_path],
                outputs=output,
            )
            _wire_stop(stop_btn, stop_status)

        with gr.Tab("Апскейл изображений (Cloud) SeedVR2"):
            with gr.Row():
                with gr.Column():
                    cloud_input_photos_path = gr.Textbox(
                        label="Путь к папке с исходными изображениями (C:\\Users\\Public...)",
                    )
                    cloud_output_result_folder = gr.Textbox(
                        label="Название папки для результатов",
                    )
                    cloud_width_scale = gr.Number(
                        label="Минимальная ширина итогового изображения",
                        value=3000,
                        step=100,
                        precision=0,
                    )
                    cloud_concurrency = gr.Slider(
                        label="Количество одновременных генераций",
                        minimum=1,
                        maximum=5,
                        value=5,
                        step=1,
                        precision=0,
                    )
                    with gr.Row():
                        cloud_run_btn = gr.Button("Upscale", variant="primary")
                        cloud_stop_btn = gr.Button("Stop", variant="stop")
                    cloud_stop_status = gr.Textbox(label="Статус остановки", interactive=False, lines=1)
                with gr.Column():
                    cloud_output = gr.Textbox(label="Результат обработки", interactive=False)

            _wire_generation(
                cloud_run_btn,
                seedvr2_batch_upscale_cloud,
                inputs=[
                    cloud_input_photos_path,
                    cloud_output_result_folder,
                    cloud_width_scale,
                    cloud_concurrency,
                ],
                outputs=cloud_output,
            )
            _wire_stop(cloud_stop_btn, cloud_stop_status)

    return demo


if __name__ == "__main__":
    if not portable_client.check_connection():
        logger.warning(
            "Cannot connect to ComfyUI Portable at %s. "
            "Make sure Portable ComfyUI is running before using local tabs.",
            config.comfyui_portable_prompt_url,
        )

    if not cloud_client.check_connection():
        logger.warning(
            "Cannot connect to ComfyUI Cloud at %s. "
            "Cloud upscale tab may not work until the service is available.",
            config.comfyui_cloud_base_url,
        )

    demo = build_ui()
    exit_code = 0

    try:
        demo.launch(
            server_name=config.ui_server_name,
            server_port=config.ui_server_port,
            share=False,
            debug=False,
            show_error=True,
            theme="soft",
        )
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception:
        logger.exception("Unexpected error during launch")
        exit_code = 1
    finally:
        demo.close()
        logger.info("Goodbye!")
        sys.exit(exit_code)
