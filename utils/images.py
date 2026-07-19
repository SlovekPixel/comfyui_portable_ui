import logging
import os
import uuid
from datetime import datetime

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_uploaded_image(image_input, dir_input: str) -> str | None:
    """Сохранение загруженного изображения с уникальным именем."""
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype("uint8"))
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"input_{timestamp}_{unique_id}.png"
    filepath = os.path.join(dir_input, filename)

    image.save(filepath)
    logger.info("Saved input image: %s (%d bytes)", filepath, os.path.getsize(filepath))

    return filepath
