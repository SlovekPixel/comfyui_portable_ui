"""
Единый механизм отмены пакетной обработки для Desktop и Cloud.

Поведение «мягкой остановки»:
- уже запущенная задача (batch / cloud job) дорабатывает до конца;
- новые задачи из очереди не стартуют;
- успешно завершённые результаты сохраняются.

Desktop (Portable): последовательные batch-циклы проверяют флаг перед submit.
Cloud: ThreadPoolExecutor подаёт новые задачи только пока флаг не установлен;
       уже выполняющиеся worker-задачи не прерываются.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")

_controller_lock = threading.Lock()
_controller: ExecutionController | None = None


class ExecutionController:
    """Потокобезопасный контроллер одного активного запуска генерации."""

    def __init__(self) -> None:
        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._running = False

    @property
    def is_running(self) -> bool:
        with self._run_lock:
            return self._running

    def begin_run(self) -> None:
        with self._run_lock:
            self._stop_event.clear()
            self._running = True

    def request_stop(self) -> bool:
        """Запрашивает остановку. Возвращает True, если генерация активна."""
        with self._run_lock:
            if not self._running:
                return False
            self._stop_event.set()
            return True

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def end_run(self) -> None:
        with self._run_lock:
            self._running = False
            self._stop_event.clear()


def get_execution_controller() -> ExecutionController:
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = ExecutionController()
    return _controller


def should_continue_processing() -> bool:
    """True — можно запускать следующую задачу из очереди."""
    return not get_execution_controller().is_stop_requested()


def request_batch_stop() -> str:
    """Обработчик кнопки Stop в Gradio."""
    if get_execution_controller().request_stop():
        return (
            "Остановка запрошена. Текущая задача завершится, "
            "новые задачи запускаться не будут."
        )
    return "Нет активной генерации."


def run_with_execution_control(handler: Callable[..., T]) -> Callable[..., T]:
    """Оборачивает workflow-handler: сброс флага при старте, очистка при выходе."""

    def wrapper(*args, **kwargs) -> T:
        controller = get_execution_controller()
        controller.begin_run()
        try:
            return handler(*args, **kwargs)
        finally:
            controller.end_run()

    return wrapper


def stopped_prefix() -> str:
    return "Остановлено пользователем.\n"
