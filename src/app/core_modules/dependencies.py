# app/api/core_modules/dependencies.py

from functools import lru_cache
from fastapi import Request

from .config import AppConfig
from .llm_integration import LLMEngine
from .retrieval import RetrievalEngine
from .prompt_engineering import PromptManager

@lru_cache
def get_config() -> AppConfig:
    """获取应用配置（单例）"""
    # 优先从环境变量加载，也可结合配置文件
    return AppConfig.from_env()

def get_llm_engine(request: Request) -> LLMEngine:
    """获取 LLM 引擎实例（从应用状态）"""
    return request.app.state.llm_engine

def get_retrieval_engine(request: Request) -> RetrievalEngine:
    return request.app.state.retrieval_engine

def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager