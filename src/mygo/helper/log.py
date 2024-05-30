import logging
from logging import Formatter, StreamHandler

# TODO: use a init_logger function instead

_formatter = Formatter(
    "%(levelname)s[%(asctime)s]: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
)
_handler = StreamHandler()
_handler.setFormatter(_formatter)

logger = logging.getLogger("mygo")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)

__all__ = ["logger"]
