import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

class _ServiceFilter(logging.Filter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service
        return True

def setup_logging(
        app_name: str = 'app',
        log_dir: str | None = None,
        retention: int = 30,
        level: int = logging.INFO,
        to_stdout: bool = True,
):
    # Main Log folder
    base_log_dir = log_dir or os.getenv("LOG_DIR", os.path.join(os.getcwd(), "logs"))
    service_log_dir = os.path.join(base_log_dir, app_name)
    os.makedirs(service_log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Make sure Handlers are not duplicated
    for h in list(root.handlers):
        root.removeHandler(h)

    log_path = os.path.join(service_log_dir, f'{datetime.now():%Y-%m-%d}.log')
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=retention,
        encoding="utf-8",
        utc=False,
    )

    # Formatter with service and name
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fmt)
    file_handler.addFilter(_ServiceFilter(app_name))
    root.addHandler(file_handler)

    if to_stdout:
        console = logging.StreamHandler()
        console.setFormatter(file_fmt)
        console.addFilter(_ServiceFilter(app_name))
        root.addHandler(console)

    root.info(f"[{app_name}] Logging initialisiert → {service_log_dir}")
    return root

def get_logger(name:str | None = None) -> logging.Logger:
    return logging.getLogger(name)