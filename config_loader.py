import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _WEB_DIR.parent

COMFYUI_CLOUD_BASE_URL_DEFAULT = "https://cloud.comfy.org"


class Config:
    """Класс для загрузки и доступа к конфигурации."""

    _instance = None
    _loaded = False
    _workflows_dir = _WEB_DIR / "api_json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_config()

    def _load_config(self) -> None:
        load_dotenv(_WEB_DIR / ".env")
        load_dotenv()
        logger.info(".env loaded (web dir: %s, cwd: %s)", _WEB_DIR, Path.cwd())
        self._validate_config()
        Config._loaded = True

    def _validate_config(self):
        required_vars = [
            "COMFYUI_PORTABLE_PROMPT_URL",
            "COMFYUI_PORTABLE_HISTORY_URL",
            "UI_SERVER_PORT",
        ]

        missing = [v for v in required_vars if not os.getenv(v)]
        if not (os.getenv("COMFYUI_CLOUD_API_KEY") or os.getenv("COMFYUI_API_KEY")):
            missing.append("COMFYUI_CLOUD_API_KEY")
        if missing:
            logger.error("Missing required environment variables:")
            for v in missing:
                logger.error("  - %s", v)
            logger.error("Please add these to your .env file in %s", _WEB_DIR)
            sys.exit(1)

        if not self._workflows_dir.exists():
            logger.error("Workflows directory '%s' not found!", self._workflows_dir)
            sys.exit(1)

        workflows = self.get_all_workflows()
        if not workflows:
            logger.warning("No .json files found in '%s'", self._workflows_dir)
        else:
            for wf in workflows:
                logger.info("Found workflow: %s", wf)

    @property
    def comfyui_portable_prompt_url(self) -> str:
        return os.getenv("COMFYUI_PORTABLE_PROMPT_URL")

    @property
    def comfyui_portable_history_url(self) -> str:
        return os.getenv("COMFYUI_PORTABLE_HISTORY_URL")

    @property
    def comfyui_cloud_base_url(self) -> str:
        return os.getenv("COMFYUI_CLOUD_BASE_URL", COMFYUI_CLOUD_BASE_URL_DEFAULT)

    @property
    def comfyui_cloud_api_key(self) -> str:
        return os.getenv("COMFYUI_CLOUD_API_KEY") or os.getenv("COMFYUI_API_KEY", "")

    @property
    def comfyui_portable_default_input_dir(self) -> str:
        return str(_PROJECT_ROOT / "ComfyUI" / "input")

    @property
    def ui_server_port(self) -> int:
        return int(os.getenv("UI_SERVER_PORT", "7860"))

    @property
    def ui_server_name(self) -> str:
        # 127.0.0.1 вместо 0.0.0.0: Gradio при 0.0.0.0 проверяет localhost,
        # что часто ломается под VPN/системным proxy (503 на startup-events).
        return os.getenv("UI_SERVER_NAME", "127.0.0.1")

    @property
    def ui_title(self) -> str:
        return os.getenv("UI_TITLE", "ComfyUI ODI")

    @property
    def workflows_dir(self) -> str:
        return str(self._workflows_dir)

    def get_workflow_path(self, workflow_name: str) -> str:
        workflow_path = self._workflows_dir / f"{workflow_name}.json"
        if not workflow_path.exists():
            logger.error("Workflow '%s' not found at %s", workflow_name, workflow_path)
            sys.exit(1)
        return str(workflow_path)

    def get_all_workflows(self) -> list[str]:
        if not self._workflows_dir.exists():
            return []
        return [f.stem for f in sorted(self._workflows_dir.glob("*.json"))]


try:
    config = Config()
    logger.info("Configuration initialized successfully")
    logger.info("Available workflows: %s", ", ".join(config.get_all_workflows()))
except SystemExit:
    raise
except Exception as e:
    logger.exception("Unexpected error during configuration loading: %s", e)
    sys.exit(1)
