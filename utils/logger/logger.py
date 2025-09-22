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
    # Main Log folder
    base_log_dir = log_dir or os.getenv("LOG_DIR", os.path.join(os.getcwd(), "logs"))

    # Each docker service gets a log folder
    service_log_dir = os.path.join(base_log_dir, app_name)
    os.makedirs(service_log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    # Make sure Handlers are not duplicated
    for h in list(logger.handlers):
        logger.removeHandler(h)

    log_path = os.path.join(service_log_dir, f'{datetime.now():%Y-%m-%d}.log')

    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=retention,
        encoding="utf-8",
        utc=False,  # falls du in der Container-Locale bleiben willst
    )
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    if to_stdout:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(console)

    logger.info(f"[{app_name}] Logging initialisiert → {service_log_dir}")
    return logger

def get_logger(name:str | None = None) -> logging.Logger:
    return logging.getLogger(name)