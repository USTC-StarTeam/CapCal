import logging
from typing import Optional
from pathlib import Path
from src.BaseConfig import LogConfig

# python -m src.logger

def get_logger(name: Optional[str] = None,log_config: Optional[LogConfig] = None) -> logging.Logger:
    """Create or retrieve a logger."""

    logger = logging.getLogger(name)
    if getattr(logger, "_bias_logger_configured", False):#如果logger已经配置过，则直接返回
        return logger

    log_config = _prepare_log_config(log_config)#确保log_config是一个LogConfig实例
    level = log_config.level
    log_file_path = log_config.resolve_path()

    logger.setLevel(level)
    logger.propagate = False

    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_format)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file_path), mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger._bias_logger_configured = True  # type: ignore[attr-defined]
    return logger

def _prepare_log_config(log_config: Optional[LogConfig]) -> LogConfig:
    """Ensure we always have a LogConfig instance."""
    if log_config is None:
        return LogConfig()
    return log_config

__all__ = ["get_logger"]