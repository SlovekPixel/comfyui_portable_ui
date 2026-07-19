import json


def format_execution_error(status: dict, execution_error: dict | None) -> str:
    error_lines = ["ComfyUI workflow завершился с ошибкой."]

    if execution_error:
        node_id = execution_error.get("node_id")
        node_type = execution_error.get("node_type")
        exception_type = execution_error.get("exception_type")
        exception_message = execution_error.get("exception_message")
        executed = execution_error.get("executed")
        current_outputs = execution_error.get("current_outputs")
        traceback = execution_error.get("traceback")

        if node_id is not None:
            error_lines.append(f"Node ID: {node_id}")

        if node_type:
            error_lines.append(f"Node: {node_type}")

        if exception_type:
            error_lines.append(f"Exception: {exception_type}")

        if exception_message:
            error_lines.append(f"Message: {exception_message.strip()}")

        if executed:
            error_lines.append(f"Executed nodes: {executed}")

        if current_outputs:
            error_lines.append(f"Current outputs: {current_outputs}")

        if traceback:
            error_lines.append("")
            error_lines.append("Traceback:")
            error_lines.extend(traceback)
    else:
        messages = status.get("messages")
        if messages:
            error_lines.append("")
            error_lines.append("Status messages:")
            error_lines.append(json.dumps(messages, indent=2, ensure_ascii=False))

    return "\n".join(error_lines)


def extract_execution_error(job: dict) -> dict | None:
    execution_error = job.get("execution_error")
    if execution_error is not None:
        return execution_error

    status = job.get("status", {})
    for message in status.get("messages", []):
        if (
            isinstance(message, list)
            and len(message) == 2
            and message[0] == "execution_error"
        ):
            return message[1]

    return None
