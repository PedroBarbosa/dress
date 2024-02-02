import sys
from loguru import logger


def setup_logger(level: int = 0, multiprocessing: bool = False):
    """
    Create loguru handler based on verbosity level provided
    """
    level_dict = {0: "INFO", 1: "DEBUG"}

    log_level = level_dict[level]

    try:
        logger.remove()
    except ValueError:
        pass

    logger.level("SUCCESS", color="<green>")
    logger.level("INFO", color="<blue>")
    logger.level("DEBUG", color="<white>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<red>")

    enqueue = True if multiprocessing else False
    log_format = "<green>{time:YYYY-MM-DD|HH:mm:ss}</green> | <lvl>{level}</lvl>: <bold>{message}</bold>"

    logger.add(
        sys.stderr,
        format=log_format,
        backtrace=False,
        colorize=True,
        level=log_level,
        enqueue=enqueue,
    )

    return logger
