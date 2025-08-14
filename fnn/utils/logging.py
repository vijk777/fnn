import logging
import time

INFO = logging.INFO
DEBUG = logging.DEBUG
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# One shared formatter instance
_UTC_FMT = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
_UTC_FMT.converter = time.gmtime  # force UTC

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger pre-wired with a console handler and UTC formatter.
    Does not touch the root logger or other libraries.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:                 # only attach once per logger
        h = logging.StreamHandler()
        h.setFormatter(_UTC_FMT)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False            # important: avoid double logging via root
    return logger
