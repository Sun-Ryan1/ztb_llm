import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class AppConfig:
    """应用全局配置"""
    env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # LLM 配置（恢复为之前的路径）
    llm_model_path: str = "/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct"
    llm_load_in_4bit: bool = True
    llm_default_max_new_tokens: int = 512
    llm_enable_cache: bool = True

    # 检索配置（恢复为之前的路径和集合名）
    chroma_db_path: str = "/tmp/chroma_db_dsw"
    embedding_model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
    collection_name: str = "rag_knowledge_base"
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.0
    rerank_enabled: bool = False
    rerank_model_path: str = "/mnt/workspace/data/modelscope/cache/bge-reranker-large/BAAI/bge-reranker-large"
    location_config_path: Optional[str] = None   # 如需要，可指定为 "./location_config.json"

    # 提示模板配置
    prompt_template_name: str = "tender_professional"
    prompt_templates_file: Optional[str] = None  # 默认不加载外部文件，使用内置模板

    # 日志
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从环境变量加载配置（前缀 APP_）"""
        config = cls()
        for key in config.__annotations__:
            env_key = f"APP_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # 尝试类型转换
                field_type = config.__annotations__[key]
                if field_type == bool:
                    value = value.lower() in ("true", "1", "yes")
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                elif field_type == Environment:
                    value = Environment(value)
                setattr(config, key, value)
        return config