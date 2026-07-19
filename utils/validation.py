import re
from pathlib import Path

import gradio as gr

from utils.constants import IMAGE_EXTENSIONS


def require_non_empty(value: str | None, message: str) -> str:
    if not value or not value.strip():
        raise gr.Error(message)
    return value.strip()


def require_existing_dir(path_str: str, not_exists_message: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise gr.Error(not_exists_message)
    return path


def list_image_files(folder: Path) -> list[Path]:
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def require_folder_name(name: str) -> str:
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise gr.Error("Название папки должно содержать только латинские буквы, цифры и _")
    return name
