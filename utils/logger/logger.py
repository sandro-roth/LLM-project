import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

def setup_logging(
        app_name: str = 'app',
        log_dir: str | None = None,
        retention: int = 30,
        level: int = logging.INFO,
        to_stdout: bool = True,
):