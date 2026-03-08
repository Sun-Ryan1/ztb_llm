# utils/logger.py

import logging
import logging.config
import os
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """
    配置全局日志

    Args:
        _level:  (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ，若为 None 则只输出到控制台
        log_format: 
        date_format: 日期格式
        max_bytes: （用于 RotatingFileHandler）
        backup_count: 
    """
    # 定义 handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        }
    }

    # 如果指定了日志文件，添加 RotatingFileHandler
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "default",
            "filename": log_file,
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }

    # 完整配置字典
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": date_format,
            },
        },
        "handlers": handlers,
        "root": {
            "handlers": list(handlers.keys()),
            "level": log_level,
        },
    }

    # 应用配置
    logging.config.dictConfig(logging_config)

    # 可选的第三方库日志级别调整（根据实际需要）
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger，自动使用配置好的格式

    Args:
        : logger 名称，通常传入 __name__

    Returns:
        .Logger 实例
    """
    return logging.getLogger(name)


# 为了方便，也可以直接提供一个初始化函数，允许从环境变量读取配置
def setup_logging_from_env() -> None:
    """从环境变量读取配置并初始化日志"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE")
    setup_logging(log_level=log_level, log_file=log_file)