from typing import Protocol


class WorkflowClient(Protocol):
    """Общий интерфейс для отправки workflow и ожидания результата."""

    def submit_workflow(self, workflow: dict) -> str:
        """Отправляет workflow и возвращает prompt_id / job_id."""

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """Ждёт завершения и возвращает outputs workflow."""

    def check_connection(self) -> bool:
        """Проверяет доступность backend."""
