import re
from typing import Iterator, Pattern, TypeVar

T = TypeVar("T")


def split_prompts_by(text: str, str_separator: str) -> list[str]:
    """
    Разбивает текст по регулярному выражению-разделителю.

    Args:
        text: Исходный текст.
        str_separator: Регулярное выражение-разделитель.

    Returns:
        Список непустых строк без лишних пробелов.
    """
    prompt_separator: Pattern[str] = re.compile(str_separator)

    return [
        part.strip()
        for part in prompt_separator.split(text)
        if part.strip()
    ]


def chunks(lst: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
