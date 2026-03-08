"""
LLM 集成模块
=============
独立的大语言模型集成模块，支持本地模型（Transformers）和 API 扩展。
提供统一的生成接口，支持批处理、缓存、量化等生产级特性。
"""

import os
import json
import time
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 配置定义 ====================

class ModelBackend(Enum):
    """支持的模型后端类型"""
    LOCAL = "local"
    API = "api"  # 预留，便于扩展


@dataclass
class LLMConfig:
    """LLM 集成模块配置"""
    # 基础配置
    backend: ModelBackend = ModelBackend.LOCAL
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_path: Optional[str] = None  # 本地路径，优先级高于 model_name
    device: Optional[str] = None  # 自动选择 cuda/cpu
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True

    # 量化配置（仅 local）
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # 推理默认参数
    default_max_new_tokens: int = 1024
    default_temperature: float = 0.1
    default_top_p: float = 0.9
    default_do_sample: bool = False
    default_repetition_penalty: float = 1.1
    default_no_repeat_ngram_size: int = 3
    use_cache: bool = True  # KV 缓存

    # 批处理
    enable_batch: bool = True
    batch_max_size: int = 8
    batch_timeout: float = 0.1  # 动态批处理等待时间（秒）

    # 缓存
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600  # 秒

    # 资源管理
    offload_folder: Optional[str] = "./offload"
    clean_cache_after_generate: bool = False  # 生成后清理显存缓存

    # API 相关（预留）
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 缓存装饰器 ====================

class LLMCache:
    """线程安全的 LLM 结果缓存"""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._lock = threading.Lock()

    def _make_key(self, prompt: str, kwargs: Dict) -> str:
        """生成缓存键（考虑关键参数）"""
        # 提取影响结果的参数（简化版，可按需扩展）
        relevant_keys = [
            "max_new_tokens", "temperature", "top_p", "do_sample",
            "repetition_penalty", "no_repeat_ngram_size"
        ]
        params = {k: kwargs.get(k) for k in relevant_keys}
        content = json.dumps([prompt, params], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, kwargs: Dict) -> Optional[str]:
        if not self._cache:
            return None
        key = self._make_key(prompt, kwargs)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                if time.time() - entry["time"] < self.ttl:
                    return entry["result"]
                else:
                    del self._cache[key]
        return None

    def set(self, prompt: str, kwargs: Dict, result: str):
        if len(self._cache) >= self.max_size:
            # 简单的 LRU 淘汰：移除最早的一个
            with self._lock:
                oldest = min(self._cache.items(), key=lambda x: x[1]["time"])
                del self._cache[oldest[0]]
        key = self._make_key(prompt, kwargs)
        with self._lock:
            self._cache[key] = {"result": result, "time": time.time()}

    def clear(self):
        with self._lock:
            self._cache.clear()


# ==================== 模型后端基类 ====================

class BaseModelBackend:
    """模型后端抽象基类"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self._is_loaded = False

    def load(self):
        """加载模型"""
        raise NotImplementedError

    def unload(self):
        """卸载模型，释放资源"""
        raise NotImplementedError

    def generate(self, prompt: str, **kwargs) -> str:
        """生成回答"""
        raise NotImplementedError

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class LocalModelBackend(BaseModelBackend):
    """本地 Transformers 模型后端"""
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None

    def _get_quant_config(self) -> Optional[BitsAndBytesConfig]:
        if self.config.load_in_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def load(self):
        if self.is_loaded:
            return
        model_path = self.config.model_path or self.config.model_name
        logger.info(f"加载本地模型: {model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=(self.config.model_path is not None)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        quantization_config = self._get_quant_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not quantization_config else None,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            offload_folder=self.config.offload_folder,
            local_files_only=(self.config.model_path is not None)
        )
        self.model.eval()
        self._is_loaded = True
        logger.info(f"模型加载成功，设备: {self.model.device}")

    def unload(self):
        if not self.is_loaded:
            return
        logger.info("卸载本地模型")
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _prepare_generation_config(self, **kwargs) -> Dict:
        """合并默认参数与传入参数"""
        params = {
            "max_new_tokens": self.config.default_max_new_tokens,
            "temperature": self.config.default_temperature,
            "top_p": self.config.default_top_p,
            "do_sample": self.config.default_do_sample,
            "repetition_penalty": self.config.default_repetition_penalty,
            "no_repeat_ngram_size": self.config.default_no_repeat_ngram_size,
            "use_cache": self.config.use_cache,
        }
        params.update(kwargs)
        return params

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded:
            self.load()
        gen_params = self._prepare_generation_config(**kwargs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_params,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True).strip()

        if self.config.clean_cache_after_generate:
            torch.cuda.empty_cache()

        return answer

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        if not self.is_loaded:
            self.load()
        if not self.config.enable_batch or len(prompts) == 1:
            return [self.generate(p, **kwargs) for p in prompts]

        gen_params = self._prepare_generation_config(**kwargs)

        # 批量编码
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_params,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 批量解码
        answers = []
        for i, output in enumerate(outputs):
            input_len = inputs.input_ids.shape[1]
            ans = self.tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            answers.append(ans)

        if self.config.clean_cache_after_generate:
            torch.cuda.empty_cache()

        return answers


class APIModelBackend(BaseModelBackend):
    """API 模型后端（预留，待实现）"""
    def load(self):
        logger.info("API 后端无需加载，直接使用")
        self._is_loaded = True

    def unload(self):
        self._is_loaded = False

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: 实现 API 调用
        raise NotImplementedError("API 后端尚未实现")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError("API 后端尚未实现")


# ==================== 主引擎类 ====================

class LLMEngine:
    """
    大语言模型统一引擎
    支持本地模型和 API 后端，提供生成、批量生成、缓存等功能。
    """
    def __init__(self, config: Union[LLMConfig, Dict]):
        if isinstance(config, dict):
            config = LLMConfig(**config)
        self.config = config

        # 选择后端
        if config.backend == ModelBackend.LOCAL:
            self.backend = LocalModelBackend(config)
        elif config.backend == ModelBackend.API:
            self.backend = APIModelBackend(config)
        else:
            raise ValueError(f"不支持的模型后端: {config.backend}")

        # 初始化缓存
        self.cache = LLMCache(max_size=config.cache_max_size, ttl=config.cache_ttl) if config.enable_cache else None

        # 加载模型
        self.backend.load()

        logger.info(f"LLMEngine 初始化完成，后端: {config.backend.value}")

    def generate(self, prompt: str, use_cache: bool = None, **kwargs) -> str:
        """
        生成回答
        :param prompt: 输入提示
        :param use_cache: 是否使用缓存（默认使用配置）
        :param kwargs: 覆盖默认推理参数
        :return: 生成文本
        """
        if use_cache is None:
            use_cache = self.config.enable_cache

        # 缓存查找
        if use_cache and self.cache:
            cached = self.cache.get(prompt, kwargs)
            if cached is not None:
                logger.debug(f"缓存命中: {prompt[:50]}...")
                return cached

        # 调用后端生成
        result = self.backend.generate(prompt, **kwargs)

        # 存入缓存
        if use_cache and self.cache:
            self.cache.set(prompt, kwargs, result)

        return result

    def batch_generate(self, prompts: List[str], use_cache: bool = None, **kwargs) -> List[str]:
        """
        批量生成
        :param prompts: 输入提示列表
        :param use_cache: 是否使用缓存
        :param kwargs: 推理参数
        :return: 生成文本列表
        """
        if use_cache is None:
            use_cache = self.config.enable_cache

        # 处理缓存
        if use_cache and self.cache:
            results = []
            uncached_prompts = []
            uncached_indices = []
            for i, p in enumerate(prompts):
                cached = self.cache.get(p, kwargs)
                if cached is not None:
                    results.append(cached)
                else:
                    uncached_prompts.append(p)
                    uncached_indices.append(i)
                    results.append(None)  # 占位

            if uncached_prompts:
                batch_results = self.backend.batch_generate(uncached_prompts, **kwargs)
                # 填充结果并缓存
                for idx, res in zip(uncached_indices, batch_results):
                    results[idx] = res
                    self.cache.set(prompts[idx], kwargs, res)
            return results
        else:
            return self.backend.batch_generate(prompts, **kwargs)

    def unload(self):
        """卸载模型，释放资源"""
        self.backend.unload()
        if self.cache:
            self.cache.clear()
        logger.info("LLMEngine 资源已释放")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 本地模型配置示例
    config = LLMConfig(
        model_path="/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct",
        load_in_4bit=True,
        default_max_new_tokens=200,
        enable_cache=True,
    )

    # 初始化引擎
    engine = LLMEngine(config)

    # 单条生成
    prompt = "请介绍一下招投标法的主要内容。"
    answer = engine.generate(prompt)
    print(f"Q: {prompt}\nA: {answer}\n")

    # 批量生成
    prompts = [
        "履约保证金的最高限额是多少？",
        "中标公告发出后发现第一名为无效投标时，招标人应如何处理？",
    ]
    answers = engine.batch_generate(prompts)
    for q, a in zip(prompts, answers):
        print(f"Q: {q}\nA: {a}\n")

    # 释放资源
    engine.unload()