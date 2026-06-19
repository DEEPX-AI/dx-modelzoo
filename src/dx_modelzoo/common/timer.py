from __future__ import annotations

from datetime import datetime

from loguru import logger


class EvaluationTimer:
    def __init__(self, debug_mode: bool = False) -> None:
        self.start_time = None
        self.debug_mode = debug_mode

    def __enter__(self):
        logger.info("Evaluation Start.")
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, *args):
        if exc_type is not None:
            import traceback

            traceback.print_exc()
        end_time = datetime.now()
        logger.info("Total RunTime {}", str(end_time - self.start_time))
        return False
