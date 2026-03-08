# src/app/api/core_modules/__init__.py

# 暴露配置类
from .config import AppConfig, Environment

# 暴露 LLM 相关
from .llm_integration import LLMEngine, LLMConfig

# 暴露检索相关
from .retrieval import RetrievalEngine, RetrievalConfig, CrossEncoderReranker

# 暴露提示工程相关
from .prompt_engineering import PromptManager, PromptTemplate, format_context

# 暴露工具函数
from .utils import extract_company_name, select_best_document

# 暴露依赖注入函数
from .dependencies import get_llm_engine, get_retrieval_engine, get_prompt_manager

__all__ = [
    # 配置
    "AppConfig",
    "Environment",

    # LLM
    "LLMEngine",
    "LLMConfig",

    # 检索
    "RetrievalEngine",
    "RetrievalConfig",
    "CrossEncoderReranker",

    # 提示工程
    "PromptManager",
    "PromptTemplate",
    "format_context",

    # 工具
    "extract_company_name",
    "select_best_document",

    # 依赖注入
    "get_llm_engine",
    "get_retrieval_engine",
    "get_prompt_manager",
]