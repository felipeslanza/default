import logging
import sys

from . import settings

logging.basicConfig(
    stream=sys.stdout,
    level=settings.LOGGING_LEVEL,
    format=settings.LOGGING_FORMAT,
)
