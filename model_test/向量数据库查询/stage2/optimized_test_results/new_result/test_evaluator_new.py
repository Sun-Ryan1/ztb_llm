#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试评估框架
"""

import os
import json
import time
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Callable
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from tqdm import tqdm
from abc import ABC, abstractmethod
import warnings
import signal

# 第三方依赖导入放在try-except中，增强兼容性
try:
    from vector_db_query_optimized import OptimizedVectorDBQuery, QueryResult
except ImportError as e:
    logging.error(f"导入向量查询模块失败: {e}")
    raise

# 配置日志（支持文件输出）
def setup_logger(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """初始化日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    # 避免重复处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s
"""指标计算器抽象基类"""
    
    @abstractmethod
    def calculate(self, expected_ids: List[str], actual_ids: List[str]) -> Dict[str, Any]:
        """计算指标"""
        pass
    
    @abstractmethod
    def get_required_k_values(self) -> List[int]:
        """返回需要计算的k值列表"""
        pass

class BaseMetricsCalculator(MetricsCalculator):
    """基础指标计算器"""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 15]
        self.k_values.sort()
    
    def get_required_k_values(self) -> List[int]:
        return self.k_values
    
    def calculate(self, expected_ids: List[str], actual_ids: List[str]) -> Dict[str, Any]:
        """计算基础指标（修复版）"""
        # 标准化ID（过滤空值）
        normalized_expected = set(filter(None, expected_ids))
        normalized_actual = list(filter(None, actual_ids))
        
        # 处理边界情况
        if not normalized_expected:
            # 无预期文档：不应计入指标统计，返回特殊标记
            return {
                "is_valid_case": False,
                "error_type": "no_expected_docs",
                "note": "无预期文档，跳过指标计算"
            }
        
        if not normalized_actual:
            # 无返回结果：所有指标为0
            result = {
                "is_valid_case": True,
                "hit_at_k": {k: False for k in self.k_values},
                "precision_at_k": {k: 0.0 for k in self.k_values},
                "recall_at_k": {k: 0.0 for k in self.k_values},
                "mrr": 0.0,
                "recall": 0.0
            }
            # 为最大k值添加额外指标
            max_k = max(self.k_values)
            result.update({
                "first_hit_rank": None,
                "hit_count": 0,
                "total_expected": len(normalized_expected)
            })
            return result
        
        # 计算各k值指标
        hit_at_k = {}
        precision_at_k = {}
        recall_at_k = {}
        
        # 计算首次命中排名和倒数排名
        first_hit_rank = None
        reciprocal_ranks = []
        hit_count = 0
        
        # 预计算命中位置
        hit_positions = []
        for idx, doc_id in enumerate(normalized_actual):
            if doc_id in normalized_expected:
                hit_positions.append(idx + 1)  # 排名从1开始
                hit_count += 1
        
        # 计算MRR
        if hit_positions:
            first_hit_rank = hit_positions[0]
            reciprocal_ranks = [1.0 / rank for rank in hit_positions]
            mrr = max(reciprocal_ranks)  # 最佳排名的倒数
        else:
            mrr = 0.0
        
        # 计算各k值指标
        for k in self.k_values:
            # 计算top_k内的命中数
            hits_in_top_k = sum(1 for rank in hit_positions if rank <= k)
            
            hit_at_k[k] = hits_in_top_k > 0
            precision_at_k[k] = hits_in_top_k / min(k, len(normalized_actual))
            recall_at_k[k] = hits_in_top_k / len(normalized_expected) if normalized_expected else 0.0
        
        # 计算总体召回率（不受k限制）
        recall = hit_count / len(normalized_expected) if normalized_expected else 0.0
        
        return {
            "is_valid_case": True,
            "hit_at_k": hit_at_k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "recall": recall,
            "first_hit_rank": first_hit_rank,
            "hit_count": hit_count,
            "total_expected": len(normalized_expected),
            "total_returned": len(normalized_actual)
        }

class EnhancedMetricsCalculator(BaseMetricsCalculator):
    """增强指标计算器（新增F1、MAP等指标）"""
    
    def calculate(self, expected_ids: List[str], actual_ids: List[str]) -> Dict[str, Any]:
        """计算增强指标"""
        base_metrics = super().calculate(expected_ids, actual_ids)
        
        if not base_metrics.get("is_valid_case", False):
            return base_metrics
        
        # 计算F1分数
        precision = base_metrics["precision_at_k"].get(10, 0.0)  # 使用P@10
        recall = base_metrics["recall"]
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 计算平均精度（AP）
        ap = self._calculate_average_precision(expected_ids, actual_ids)
        
        # 计算NDCG（归一化折损累计增益）
        ndcg = self._calculate_ndcg(expected_ids, actual_ids, max_k=10)
        
        base_metrics.update({
            "f1_score": f1_score,
            "average_precision": ap,
            "ndcg_at_10": ndcg,
            "hit_positions": self._get_hit_positions(expected_ids, actual_ids)
        })
        
        return base_metrics
    
    def _calculate_average_precision(self, expected_ids: List[str], actual_ids: List[str]) -> float:
        """计算平均精度（AP）"""
        normalized_expected = set(filter(None, expected_ids))
        normalized_actual = list(filter(None, actual_ids))
        
        if not normalized_expected or not normalized_actual:
            return 0.0
        
        precision_values = []
        relevant_count = 0
        
        for i, doc_id in enumerate(normalized_actual, 1):
            if doc_id in normalized_expected:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_values.append(precision_at_i)
        
        if not precision_values:
            return 0.0
        
        return sum(precision_values) / min(len(normalized_expected), len(actual_ids))
    
    def _calculate_ndcg(self, expected_ids: List[str], actual_ids: List[str], max_k: int = 10) -> float:
        """计算NDCG@k"""
        normalized_expected = set(filter(None, expected_ids))
        normalized_actual = list(filter(None, actual_ids))[:max_k]
        
        if not normalized_expected or not normalized_actual:
            return 0.0
        
        # 计算DCG
        dcg = 0.0
        for i, doc_id in enumerate(normalized_actual, 1):
            if doc_id in normalized_expected:
                # 相关文档增益
                gain = 1.0  # 二值相关度
                dcg += gain / np.log2(i + 1)  # 对数折扣
        
        # 计算IDCG（理想情况下的DCG）
        idcg = 0.0
        ideal_relevant_count = min(len(normalized_expected), max_k)
        for i in range(1, ideal_relevant_count + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _get_hit_positions(self, expected_ids: List[str], actual_ids: List[str]) -> List[int]:
        """获取命中位置"""
        normalized_expected = set(filter(None, expected_ids))
        normalized_actual = list(filter(None, actual_ids))
        
        hit_positions = []
        for i, doc_id in enumerate(normalized_actual, 1):
            if doc_id in normalized_expected:
                hit_positions.append(i)
        
        return hit_positions

@dataclass
class TestResult:
    """测试结果数据类（优化版）"""
    test_id: str
    query: str
    expected_doc_ids: List[str]
    actual_doc_ids: List[str]
    query_type: str
    difficulty: str
    retrieval_method: str
    retrieval_time: float
    top_k: int
    metrics: Dict[str, Any]  # 包含所有指标计算结果
    retrieved_docs: List[Dict] = field(default_factory=list)
    is_failed: bool = False
    error_msg: str = ""
    warning_msgs: List[str] = field(default_factory=list)  # 新增：警告信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 新增：元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（支持序列化）"""
        result = {
            "test_id": self.test_id,
            "query": self.query,
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "retrieval_method": self.retrieval_method,
            "retrieval_time": self.retrieval_time,
            "expected_docs": len(self.expected_doc_ids),
            "retrieved_docs": len(self.actual_doc_ids),
            "is_failed": self.is_failed,
            "error_msg": self.error_msg,
            "warning_msgs": self.warning_msgs,
            "has_valid_metrics": self.metrics.get("is_valid_case", False),
            **self.metrics
        }
        
        # 添加文档预览（安全处理）
        if self.retrieved_docs:
            result["retrieved_docs_preview"] = [
                {
                    "id": doc.get("id", ""),
                    "content_preview": str(doc.get("content_preview", ""))[:200],
                    "similarity": float(doc.get("similarity", 0.0))
                }
                for doc in self.retrieved_docs[:3]
            ]
        
        return result

class OptimizedTestEvaluator:
    """优化版测试评估器（专业版）"""
    
    DEFAULT_CONFIG = {
        "max_test_cases": None,
        "k_values": [1, 3, 5, 10, 15],
        "enable_progress_bar": True,
        "top_k": 20,
        "normalize_doc_id_lowercase": True,
        "overwrite_output": False,
        "log_dir": "logs",
        "metrics_calculator": "enhanced",  # "basic" 或 "enhanced"
        "save_detailed_results": True,
        "save_failed_cases_only": False,
        "enable_warnings": True,
        "timeout_per_case": 30.0,  # 单用例超时时间（秒）
        "retry_failed_cases": False,  # 是否重试失败用例
        "max_retries": 2,  # 最大重试次数
        "batch_size": 10,  # 批量处理大小（用于进度更新）
        "calculate_similarity_stats": True,  # 是否计算相似度统计
    }
    
    def __init__(self, 
                 test_data_path: str, 
                 output_dir: str = "optimized_test_results",
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化评估器
        :param test_data_path: 测试数据文件路径
        :param output_dir: 输出目录
        :param config: 自定义配置（覆盖默认配置）
        """
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # 合并配置并校验
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._validate_config()
        
        # 创建必要目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.config["log_dir"], exist_ok=True)
        
        # 初始化指标计算器
        self.metrics_calculator = self._init_metrics_calculator()
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        self._validate_test_data()
        
        logger.info(f"✅ 加载测试数据完成: 共{len(self.test_data['test_cases'])}个测试用例")
        logger.info(f"📊 使用指标计算器: {self.config['metrics_calculator']}")
    
    def _validate_config(self) -> None:
        """校验配置参数合法性"""
        # 校验k_values
        if not isinstance(self.config["k_values"], list) or not all(isinstance(k, int) and k > 0 for k in self.config["k_values"]):
            raise ValueError("k_values必须是正整数列表")
        
        # 校验top_k
        if not isinstance(self.config["top_k"], int) or self.config["top_k"] <= 0:
            raise ValueError("top_k必须是正整数")
        
        # 校验max_test_cases
        if self.config["max_test_cases"] is not None and (not isinstance(self.config["max_test_cases"], int) or self.config["max_test_cases"] <= 0):
            raise ValueError("max_test_cases必须是None或正整数")
        
        # 校验布尔参数
        bool_keys = ["enable_progress_bar", "normalize_doc_id_lowercase", "overwrite_output", 
                    "save_detailed_results", "save_failed_cases_only", "enable_warnings",
                    "retry_failed_cases", "calculate_similarity_stats"]
        for key in bool_keys:
            if not isinstance(self.config[key], bool):
                raise ValueError(f"{key}必须是布尔值")
        
        # 校验超时时间
        if self.config["timeout_per_case"] <= 0:
            raise ValueError("timeout_per_case必须大于0")
        
        # 校验重试次数
        if not isinstance(self.config["max_retries"], int) or self.config["max_retries"] < 0:
            raise ValueError("max_retries必须是非负整数")
        
        # 校验批量大小
        if not isinstance(self.config["batch_size"], int) or self.config["batch_size"] <= 0:
            raise ValueError("batch_size必须是正整数")
    
    def _init_metrics_calculator(self) -> MetricsCalculator:
        """初始化指标计算器"""
        calculator_type = self.config["metrics_calculator"]
        
        if calculator_type == "basic":
            return BaseMetricsCalculator(self.config["k_values"])
        elif calculator_type == "enhanced":
            return EnhancedMetricsCalculator(self.config["k_values"])
        else:
            logger.warning(f"未知的计算器类型: {calculator_type}，使用默认增强版")
            return EnhancedMetricsCalculator(self.config["k_values"])
    
    def load_test_data(self) -> Dict[str, Any]:
        """加载并校验测试数据"""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 基础校验
            if "test_cases" not in data or not isinstance(data["test_cases"], list):
                raise ValueError("测试数据必须包含test_cases列表")
            
            # 校验元数据
            metadata = data.get("metadata", {})
            if "created_at" not in metadata:
                metadata["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if "total_cases" not in metadata:
                metadata["total_cases"] = len(data["test_cases"])
            
            data["metadata"] = metadata
            return data
            
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            raise
    
    def _validate_test_data(self) -> None:
        """验证测试数据质量"""
        test_cases = self.test_data["test_cases"]
        
        # 统计信息
        stats = {
            "total": len(test_cases),
            "has_empty_query": 0,
            "has_empty_expected": 0,
            "has_invalid_expected": 0,
            "unique_query_types": set(),
            "unique_difficulties": set()
        }
        
        for idx, case in enumerate(test_cases):
            # 检查必填字段
            required_fields = ["test_id", "query", "expected_doc_ids"]
            missing_fields = [f for f in required_fields if f not in case]
            if missing_fields:
                logger.warning(f"测试用例{idx}缺少必填字段: {missing_fields}")
            
            # 检查查询文本
            query = case.get("query", "")
            if not query or not query.strip():
                stats["has_empty_query"] += 1
                logger.warning(f"测试用例{idx}查询文本为空")
            
            # 检查预期文档
            expected_docs = case.get("expected_doc_ids", [])
            if not expected_docs:
                stats["has_empty_expected"] += 1
                logger.warning(f"测试用例{idx}预期文档列表为空")
            else:
                # 检查预期文档是否都是字符串
                invalid_docs = [doc for doc in expected_docs if not isinstance(doc, (str, int))]
                if invalid_docs:
                    stats["has_invalid_expected"] += 1
                    logger.warning(f"测试用例{idx}有无效预期文档类型: {invalid_docs[:3]}")
            
            # 收集查询类型和难度
            stats["unique_query_types"].add(case.get("query_type", "unknown"))
            stats["unique_difficulties"].add(case.get("difficulty", "medium"))
        
        # 输出统计信息
        logger.info(f"📋 测试数据质量统计:")
        logger.info(f"   总用例数: {stats['total']}")
        logger.info(f"   空查询用例: {stats['has_empty_query']}")
        logger.info(f"   无预期文档用例: {stats['has_empty_expected']}")
        logger.info(f"   无效预期文档用例: {stats['has_invalid_expected']}")
        logger.info(f"   查询类型: {len(stats['unique_query_types'])}种")
        logger.info(f"   难度等级: {len(stats['unique_difficulties'])}种")
        
        # 建议
        if stats["has_empty_expected"] > stats["total"] * 0.1:
            logger.warning("⚠️  超过10%的用例无预期文档，可能影响评估准确性")
    
    def _normalize_doc_id(self, doc_id: Union[str, int, None]) -> str:
        """
        标准化文档ID（增强鲁棒性）
        :param doc_id: 原始文档ID（支持字符串/整数/None）
        :return: 标准化后的ID
        """
        if doc_id is None:
            return ""
        
        # 统一转为字符串
        if not isinstance(doc_id, str):
            doc_id = str(doc_id)
        
        # 基础处理
        doc_id = doc_id.strip()
        
        # 可选转小写
        if self.config["normalize_doc_id_lowercase"]:
            doc_id = doc_id.lower()
        
        return doc_id
    
    @contextmanager
    def _timeout_context(self, timeout: float, test_id: str = ""):
        """超时上下文管理器"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"测试用例{test_id}执行超时 ({timeout}秒)")
        
        # 设置信号处理（仅限Unix系统）
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            yield
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 取消超时
    
    def _execute_test_case(self, query_engine: OptimizedVectorDBQuery, 
                          test_case: Dict[str, Any], 
                          retry_count: int = 0) -> TestResult:
        """执行单个测试用例（带重试机制）"""
        test_id = test_case.get("test_id", "unknown")
        
        try:
            # 基础信息提取
            query = test_case.get("query", "")
            expected_doc_ids = test_case.get("expected_doc_ids", [])
            query_type = test_case.get("query_type", "general")
            difficulty = test_case.get("difficulty", "medium")
            
            # 执行查询
            start_time = time.time()
            
            # 应用超时保护
            try:
                if hasattr(signal, 'SIGALRM'):  # Unix系统支持信号
                    with self._timeout_context(self.config["timeout_per_case"], test_id):
                        query_result = query_engine.query(
                            query_text=query,
                            top_k=self.config["top_k"],
                            return_format="structured"
                        )
                else:
                    query_result = query_engine.query(
                        query_text=query,
                        top_k=self.config["top_k"],
                        return_format="structured"
                    )
            except TimeoutError as e:
                raise e
            except Exception as e:
                # 查询执行失败，尝试重试
                if self.config["retry_failed_cases"] and retry_count < self.config["max_retries"]:
                    logger.warning(f"测试用例{test_id}查询失败，第{retry_count+1}次重试: {e}")
                    return self._execute_test_case(query_engine, test_case, retry_count + 1)
                else:
                    raise
            
            retrieval_time = time.time()
"""计算相似度统计"""if not retrieved_docs:
            return {}
        
        similarities = [doc.get("similarity", 0.0) for doc in retrieved_docs if doc.get("similarity") is not None]
        
        if not similarities:
            return {}
        
        return {
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "q1": float(np.percentile(similarities, 25)),
            "q3": float(np.percentile(similarities, 75)),
            "count": len(similarities)
        }
    
    def _create_failed_test_result(self, test_case: Dict[str, Any], 
                                  error_msg: str, 
                                  is_timeout: bool = False) -> TestResult:
        """创建失败用例的结果对象"""
        # 初始化空的指标
        empty_metrics = {
            "is_valid_case": False,
            "error_type": "timeout" if is_timeout else "execution_error",
            "note": error_msg
        }
        
        return TestResult(
            test_id=test_case.get("test_id", "unknown"),
            query=test_case.get("query", ""),
            expected_doc_ids=test_case.get("expected_doc_ids", []),
            actual_doc_ids=[],
            query_type=test_case.get("query_type", "general"),
            difficulty=test_case.get("difficulty", "medium"),
            retrieval_method="failed",
            retrieval_time=0.0,
            top_k=self.config["top_k"],
            metrics=empty_metrics,
            retrieved_docs=[],
            is_failed=True,
            error_msg=error_msg,
            warning_msgs=[],
            metadata={
                "is_timeout": is_timeout,
                "retry_count": 0
            }
        )
    
    def run_test(self, query_engine: OptimizedVectorDBQuery) -> List[TestResult]:
        """
        运行测试（批量处理，增强性能）
        :param query_engine: 向量查询引擎实例
        :return: 所有测试结果（含失败用例）
        """
        test_cases = self.test_data["test_cases"]
        
        # 限制测试用例数量
        if self.config["max_test_cases"] is not None:
            test_cases = test_cases[:self.config["max_test_cases"]]
        
        total_cases = len(test_cases)
        logger.info(f"🚀 开始执行测试: 共{total_cases}个测试用例")
        
        results = []
        
        # 使用进度条
        if self.config["enable_progress_bar"]:
            pbar = tqdm(total=total_cases, desc="测试进度", unit="case")
        
        # 批量处理
        batch_size = self.config["batch_size"]
        
        try:
            for batch_start in range(0, total_cases, batch_size):
                batch_end = min(batch_start + batch_size, total_cases)
                batch_cases = test_cases[batch_start:batch_end]
                
                batch_results = []
                for test_case in batch_cases:
                    try:
                        result = self._execute_test_case(query_engine, test_case)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"批量处理中执行用例失败: {e}")
                        failed_result = self._create_failed_test_result(
                            test_case, f"批量处理异常: {str(e)}"
                        )
                        batch_results.append(failed_result)
                
                results.extend(batch_results)
                
                # 更新进度
                if self.config["enable_progress_bar"]:
                    success_count = len([r for r in results if not r.is_failed])
                    fail_count = len([r for r in results if r.is_failed])
                    avg_time = np.mean([r.retrieval_time for r in results if not r.is_failed]) if success_count > 0 else 0
                    
                    pbar.update(len(batch_results))
                    pbar.set_postfix({
                        "成功": success_count,
                        "失败": fail_count,
                        "平均耗时": f"{avg_time:.3f}s"
                    })
        
        finally:
            if self.config["enable_progress_bar"]:
                pbar.close()
        
        # 统计结果
        success_count = len([r for r in results if not r.is_failed])
        fail_count = len([r for r in results if r.is_failed])
        logger.info(f"✅ 测试执行完成: 成功{success_count}个, 失败{fail_count}个")
        
        # 详细统计
        if results:
            self._log_detailed_statistics(results)
        
        return results
    
    def _log_detailed_statistics(self, results: List[TestResult]):
        """输出详细统计信息"""
        successful_results = [r for r in results if not r.is_failed]
        
        if not successful_results:
            logger.warning("无成功用例，跳过详细统计")
            return
        
        # 按查询类型统计
        query_type_stats = defaultdict(list)
        for r in successful_results:
            query_type_stats[r.query_type].append(r)
        
        logger.info("📊 按查询类型统计:")
        for qtype, cases in sorted(query_type_stats.items(), key=lambda x: len(x[1]), reverse=True):
            avg_time = np.mean([r.retrieval_time for r in cases])
            hit_at_1 = np.mean([1 if r.metrics.get("hit_at_k", {}).get(1, False) else 0 for r in cases])
            logger.info(f"  {qtype}: {len(cases)}例 | Hit@1: {hit_at_1*100:.1f}% | 平均耗时: {avg_time:.3f}s")
        
        # 按检索方法统计
        method_stats = defaultdict(list)
        for r in successful_results:
            method_stats[r.retrieval_method].append(r)
        
        logger.info("🔧 按检索方法统计:")
        for method, cases in sorted(method_stats.items(), key=lambda x: len(x[1]), reverse=True):
            avg_time = np.mean([r.retrieval_time for r in cases])
            hit_at_1 = np.mean([1 if r.metrics.get("hit_at_k", {}).get(1, False) else 0 for r in cases])
            logger.info(f"  {method}: {len(cases)}例 | Hit@1: {hit_at_1*100:.1f}% | 平均耗时: {avg_time:.3f}s")
    
    def _get_safe_file_path(self, filename: str, ext: str = "json") -> str:
        """
        获取安全的文件路径（避免覆盖）
        :param filename: 基础文件名
        :param ext: 文件扩展名
        :return: 安全的文件路径
        """
        base_path = os.path.join(self.output_dir, f"{filename}.{ext}")
        
        # 如果允许覆盖，直接返回
        if self.config["overwrite_output"]:
            return base_path
        
        # 避免覆盖：添加时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_path = os.path.join(self.output_dir, f"{filename}_{timestamp}.{ext}")
        
        return safe_path
    
    def save_results(self, results: List[TestResult], filename: str = "test_results") -> Dict[str, Any]:
        """
        保存结果（增强容错，多种格式）
        :param results: 测试结果列表
        :param filename: 基础文件名
        :return: 统计信息
        """
        if not results:
            logger.warning("⚠️ 无测试结果可保存")
            return {}
        
        try:
            # 1. 计算统计信息
            stats = self.calculate_statistics(results)
            
            # 2. 保存统计信息
            stats_path = self._get_safe_file_path(f"{filename}_stats")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"📈 统计信息已保存: {stats_path}")
            
            # 3. 保存详细结果（可选）
            if self.config["save_detailed_results"]:
                # 过滤失败用例
                if self.config["save_failed_cases_only"]:
                    results_to_save = [r for r in results if r.is_failed]
                    save_type = "失败用例"
                else:
                    results_to_save = results
                    save_type = "全部"
                
                if results_to_save:
                    detailed_path = self._get_safe_file_path(f"{filename}_detailed")
                    simplified_results = [r.to_dict() for r in results_to_save]
                    
                    with open(detailed_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "metadata": {
                                "total_cases": len(results),
                                "saved_cases": len(results_to_save),
                                "save_type": save_type,
                                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                            },
                            "results": simplified_results
                        }, f, ensure_ascii=False, indent=2)
                    logger.info(f"📄 详细结果已保存 ({save_type}): {detailed_path}")
            
            # 4. 生成并保存分析报告
            analysis = self.generate_analysis(results, stats)
            analysis_path = self._get_safe_file_path(f"{filename}_analysis")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"📊 分析报告已保存: {analysis_path}")
            
            # 5. 保存CSV格式（便于Excel分析）
            csv_path = self._get_safe_file_path(f"{filename}_summary", "csv")
            self._save_results_to_csv(results, csv_path)
            logger.info(f"📋 CSV格式已保存: {csv_path}")
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
            raise
    
    def _save_results_to_csv(self, results: List[TestResult], csv_path: str):
        """保存结果为CSV格式"""
        # 提取关键字段
        rows = []
        for r in results:
            row = {
                "test_id": r.test_id,
                "query_type": r.query_type,
                "difficulty": r.difficulty,
                "retrieval_method": r.retrieval_method,
                "retrieval_time": r.retrieval_time,
                "expected_docs": len(r.expected_doc_ids),
                "retrieved_docs": len(r.actual_doc_ids),
                "is_failed": r.is_failed,
                "error_msg": r.error_msg[:100] if r.error_msg else "",
                "has_valid_metrics": r.metrics.get("is_valid_case", False)
            }
            
            # 添加指标
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                metrics = r.metrics
                row.update({
                    "hit_at_1": metrics.get("hit_at_k", {}).get(1, False),
                    "hit_at_3": metrics.get("hit_at_k", {}).get(3, False),
                    "hit_at_5": metrics.get("hit_at_k", {}).get(5, False),
                    "precision_at_5": metrics.get("precision_at_k", {}).get(5, 0.0),
                    "recall_at_5": metrics.get("recall_at_k", {}).get(5, 0.0),
                    "mrr": metrics.get("mrr", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "hit_count": metrics.get("hit_count", 0),
                    "first_hit_rank": metrics.get("first_hit_rank", None)
                })
            
            rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    def calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        计算统计信息（专业版）
        :param results: 测试结果列表
        :return: 统计字典
        """
        if not results:
            return {}
        
        # 过滤成功用例
        successful_results = [r for r in results if not r.is_failed]
        total_count = len(results)
        success_count = len(successful_results)
        fail_count = total_count
"""
        生成分析报告（专业版）
        :param results: 测试结果列表
        :param stats: 统计信息
        :return: 分析报告
        """analysis = {
            "overview": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": stats["summary"]["total_test_cases"],
                "success_rate": stats["summary"]["success_rate"],
                "valid_case_rate": stats["summary"]["valid_case_rate"],
                "key_performance": {
                    "avg_retrieval_time": stats.get("performance", {}).get("avg_retrieval_time", 0.0),
                    "hit_at_1": stats.get("performance", {}).get("hit_at_1", 0.0),
                    "avg_mrr": stats.get("performance", {}).get("avg_mrr", 0.0),
                    "avg_recall": stats.get("performance", {}).get("avg_recall", 0.0)
                }
            },
            "issues": [],
            "recommendations": [],
            "detailed_analysis": {
                "by_query_type": {},
                "by_difficulty": {},
                "by_retrieval_method": {},
                "failed_cases_analysis": {},
                "performance_bottlenecks": {}
            }
        }
        
        # 1. 识别核心问题
        performance = stats.get("performance", {})
        
        # 命中率问题
        hit_at_1 = performance.get("hit_at_1", 0.0)
        if hit_at_1 < 0.2:
            analysis["issues"].append(f"❌ 致命问题: Hit@1命中率极低({hit_at_1*100:.1f}%)，检索系统基本失效")
        elif hit_at_1 < 0.4:
            analysis["issues"].append(f"⚠️ 严重问题: Hit@1命中率较低({hit_at_1*100:.1f}%)，检索准确性不足")
        elif hit_at_1 < 0.6:
            analysis["issues"].append(f"⚠️ 一般问题: Hit@1命中率一般({hit_at_1*100:.1f}%)，有提升空间")
        
        # 性能问题
        avg_time = performance.get("avg_retrieval_time", 0.0)
        if avg_time > 2.0:
            analysis["issues"].append(f"❌ 致命问题: 平均查询时间过长({avg_time:.2f}s)，性能严重不足")
        elif avg_time > 1.0:
            analysis["issues"].append(f"⚠️ 严重问题: 平均查询时间偏长({avg_time:.2f}s)，影响用户体验")
        elif avg_time > 0.5:
            analysis["issues"].append(f"⚠️ 一般问题: 平均查询时间稍长({avg_time:.2f}s)，可考虑优化")
        
        # 空结果问题
        empty_rate = performance.get("empty_result_rate", 0.0)
        if empty_rate > 0.3:
            analysis["issues"].append(f"❌ 致命问题: 空结果占比过高({empty_rate*100:.1f}%)，数据或查询逻辑有问题")
        elif empty_rate > 0.1:
            analysis["issues"].append(f"⚠️ 严重问题: 空结果占比较高({empty_rate*100:.1f}%)，需检查检索逻辑")
        
        # 失败率问题
        fail_rate = stats["summary"]["fail_rate"]
        if fail_rate > 0.2:
            analysis["issues"].append(f"❌ 致命问题: 测试用例失败率过高({fail_rate*100:.1f}%)，代码稳定性差")
        elif fail_rate > 0.05:
            analysis["issues"].append(f"⚠️ 严重问题: 测试用例失败率偏高({fail_rate*100:.1f}%)，需修复")
        
        # 2. 按查询类型分析
        query_type_analysis = stats.get("by_query_type", {})
        worst_query_types = []
        
        for qtype, data in query_type_analysis.items():
            if data.get("hit_at_1", 0.0) < 0.3:
                worst_query_types.append({
                    "type": qtype,
                    "hit_at_1": data["hit_at_1"],
                    "count": data["count"],
                    "issue": f"{qtype}类型查询命中率过低"
                })
        
        if worst_query_types:
            analysis["detailed_analysis"]["worst_performing_types"] = sorted(
                worst_query_types, 
                key=lambda x: x["hit_at_1"]
            )[:5]  # 只显示最差的5种
        
        # 3. 按检索方法分析
        method_stats = defaultdict(lambda: {
            "count": 0, "hit_at_1": 0.0, "avg_time": 0.0, "avg_mrr": 0.0
        })
        
        for r in results:
            if r.is_failed or not r.metrics.get("is_valid_case", False):
                continue
            
            method = r.retrieval_method
            method_stats[method]["count"] += 1
            method_stats[method]["hit_at_1"] += 1 if r.metrics.get("hit_at_k", {}).get(1, False) else 0
            method_stats[method]["avg_time"] += r.retrieval_time
            method_stats[method]["avg_mrr"] += r.metrics.get("mrr", 0.0)
        
        # 计算平均值
        for method, data in method_stats.items():
            if data["count"] > 0:
                data["hit_at_1"] /= data["count"]
                data["avg_time"] /= data["count"]
                data["avg_mrr"] /= data["count"]
        
        analysis["detailed_analysis"]["by_retrieval_method"] = dict(method_stats)
        
        # 4. 失败用例分析
        failed_cases = [r for r in results if r.is_failed]
        if failed_cases:
            error_types = defaultdict(int)
            timeout_cases = []
            
            for r in failed_cases:
                error_type = r.metadata.get("is_timeout", False) and "TimeoutError" or r.error_msg.split(":")[0]
                error_types[error_type] += 1
                
                if r.metadata.get("is_timeout", False):
                    timeout_cases.append(r.test_id)
            
            analysis["detailed_analysis"]["failed_cases_analysis"] = {
                "total_failed": len(failed_cases),
                "error_distribution": dict(error_types),
                "timeout_cases": timeout_cases[:10],  # 只显示前10个超时用例
                "failure_rate_by_query_type": self._calculate_failure_rate_by_type(results)
            }
        
        # 5. 性能瓶颈分析
        if performance:
            time_p95 = performance.get("retrieval_time_p95", 0.0)
            time_p99 = performance.get("retrieval_time_p99", 0.0)
            
            if time_p95 > avg_time * 2:
                analysis["detailed_analysis"]["performance_bottlenecks"]["p95_high"] = {
                    "p95_time": time_p95,
                    "avg_time": avg_time,
                    "ratio": time_p95 / avg_time if avg_time > 0 else 0,
                    "insight": "部分查询耗时远高于平均值，可能存在性能瓶颈"
                }
            
            slow_cases = sorted([r for r in results if not r.is_failed], 
                               key=lambda x: x.retrieval_time, 
                               reverse=True)[:5]
            
            analysis["detailed_analysis"]["performance_bottlenecks"]["slowest_cases"] = [
                {
                    "test_id": r.test_id,
                    "query_type": r.query_type,
                    "retrieval_method": r.retrieval_method,
                    "time": r.retrieval_time,
                    "query_preview": r.query[:50] + "..."
                }
                for r in slow_cases
            ]
        
        # 6. 生成优化建议
        self._generate_recommendations(analysis, stats, results)
        
        return analysis
    
    def _calculate_failure_rate_by_type(self, results: List[TestResult]) -> Dict[str, float]:
        """计算各查询类型的失败率"""
        type_stats = defaultdict(lambda: {"total": 0, "failed": 0})
        
        for r in results:
            qtype = r.query_type
            type_stats[qtype]["total"] += 1
            if r.is_failed:
                type_stats[qtype]["failed"] += 1
        
        failure_rates = {}
        for qtype, stats in type_stats.items():
            if stats["total"] > 0:
                failure_rates[qtype] = stats["failed"] / stats["total"]
        
        return dict(sorted(failure_rates.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def _generate_recommendations(self, analysis: Dict[str, Any], stats: Dict[str, Any], results: List[TestResult]) -> None:
        """生成针对性优化建议"""
        performance = stats.get("performance", {})
        summary = stats["summary"]
        
        # 通用建议
        analysis["recommendations"].extend([
            "📊 定期运行测试评估，建立性能基线",
            "🔧 实现持续集成，每次代码变更都运行核心测试集",
            "📈 建立监控系统，实时跟踪生产环境检索性能",
            "🧪 增加压力测试，模拟高并发场景",
            "📚 建立知识库，记录常见问题和解决方案"
        ])
        
        # 命中率优化建议
        hit_at_1 = performance.get("hit_at_1", 0.0)
        if hit_at_1 < 0.5:
            analysis["recommendations"].extend([
                "🎯 优化embedding模型：尝试更大尺寸或专门训练的模型",
                "🔍 增强查询理解：添加查询扩展、同义词替换",
                "📄 改进文档处理：优化分块策略和元数据提取",
                "🔄 引入重排序：使用交叉编码器对初筛结果重排序",
                "🏷️ 加强元数据过滤：对特殊查询类型实施精确匹配"
            ])
        
        # 性能优化建议
        avg_time = performance.get("avg_retrieval_time", 0.0)
        if avg_time > 0.5:
            analysis["recommendations"].extend([
                "⚡ 优化向量索引：调整HNSW参数(m=32, ef_construction=400)",
                "💾 启用智能缓存：缓存高频查询和向量编码结果",
                "🖥️ 硬件加速：使用GPU推理、模型量化",
                "🧮 计算优化：减少不必要的相似度计算",
                "📦 批量处理：对批量查询采用异步处理"
            ])
        
        # 失败率优化建议
        fail_rate = summary.get("fail_rate", 0.0)
        if fail_rate > 0.1:
            analysis["recommendations"].extend([
                "🐛 修复已知缺陷：针对失败用例逐一排查",
                "🛡️ 增强异常处理：添加更全面的try-catch和参数校验",
                "🔁 实现重试机制：对网络超时等临时失败自动重试",
                "📝 完善日志记录：为失败用例添加上下文信息",
                "🧪 增加边界测试：覆盖极端情况和边界条件"
            ])
        
        # 空结果优化建议
        empty_rate = performance.get("empty_result_rate", 0.0)
        if empty_rate > 0.1:
            analysis["recommendations"].extend([
                "🔎 检查数据覆盖：确保测试查询在知识库中有相关内容",
                "🧠 优化查询改写：对无结果查询进行智能改写",
                "📋 添加兜底策略：无结果时返回相关或通用内容",
                "📊 分析查询模式：识别高频无结果查询类型"
            ])
    
    def print_summary(self, stats: Dict[str, Any], analysis: Dict[str, Any]):
        """打印格式化的总结报告（增强版）"""
        print("\n" + "="*100)
        print("📊 向量检索测试评估总结报告")
        print("="*100)
        
        # 基础统计
        summary = stats.get("summary", {})
        print(f"\n🔍 测试概览:")
        print(f"   总测试用例数: {summary.get('total_test_cases', 0)}")
        print(f"   成功执行数: {summary.get('successful_cases', 0)} ({summary.get('success_rate', 0)*100:.1f}%)")
        print(f"   执行失败数: {summary.get('failed_cases', 0)} ({summary.get('fail_rate', 0)*100:.1f}%)")
        print(f"   有效指标用例: {summary.get('valid_metric_cases', 0)} ({summary.get('valid_case_rate', 0)*100:.1f}%)")
        
        # 性能指标
        performance = stats.get("performance", {})
        if performance:
            print(f"\n⚡ 性能指标:")
            print(f"   平均查询耗时: {performance.get('avg_retrieval_time', 0):.3f}s")
            print(f"   P95查询耗时: {performance.get('retrieval_time_p95', 0):.3f}s")
            print(f"   空结果占比: {performance.get('empty_result_rate', 0)*100:.1f}%")
            
            if performance.get("avg_similarity"):
                print(f"   平均相似度: {performance.get('avg_similarity', 0):.3f}")
            
            print(f"\n🎯 准确性指标:")
            for k in [1, 3, 5, 10]:
                hit_rate = performance.get(f"hit_at_{k}", 0) * 100
                precision = performance.get(f"precision_at_{k}", 0) * 100
                recall = performance.get(f"recall_at_{k}", 0) * 100
                print(f"   Hit@{k}: {hit_rate:5.1f}% | 精确率@{k}: {precision:5.1f}% | 召回率@{k}: {recall:5.1f}%")
            
            print(f"\n📈 高级指标:")
            print(f"   平均MRR: {performance.get('avg_mrr', 0)*100:5.1f}%")
            print(f"   平均召回率: {performance.get('avg_recall', 0)*100:5.1f}%")
            print(f"   平均F1分数: {performance.get('avg_f1_score', 0)*100:5.1f}%")
            print(f"   平均NDCG@10: {performance.get('avg_ndcg_at_10', 0)*100:5.1f}%")
            
            if performance.get("avg_first_hit_rank"):
                print(f"   平均首次命中排名: {performance.get('avg_first_hit_rank', 0):.1f}")
                print(f"   中位数首次命中排名: {performance.get('median_first_hit_rank', 0):.1f}")
        
        # 按查询类型统计
        query_type_stats = stats.get("by_query_type", {})
        if query_type_stats:
            print(f"\n📋 按查询类型统计（前5种）:")
            sorted_qtypes = sorted(query_type_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            for qtype, data in sorted_qtypes:
                hit_rate = data.get("hit_at_1", 0) * 100
                print(f"   {qtype:<15} 用例数: {data['count']:<4} | Hit@1: {hit_rate:5.1f}% | "
                      f"平均耗时: {data.get('avg_time', 0):.3f}s | 平均MRR: {data.get('avg_mrr', 0)*100:5.1f}%")
        
        # 按检索方法统计
        method_analysis = analysis["detailed_analysis"].get("by_retrieval_method", {})
        if method_analysis:
            print(f"\n🔧 按检索方法统计:")
            for method, data in method_analysis.items():
                hit_rate = data.get("hit_at_1", 0) * 100
                print(f"   {method:<20} 用例数: {data['count']:<4} | Hit@1: {hit_rate:5.1f}% | "
                      f"平均耗时: {data.get('avg_time', 0):.3f}s")
        
        # 问题和建议
        if analysis.get("issues"):
            print(f"\n⚠️ 发现的核心问题:")
            for idx, issue in enumerate(analysis["issues"][:5], 1):  # 只显示前5个问题
                print(f"   {idx}. {issue}")
        
        if analysis.get("recommendations"):
            print(f"\n💡 关键优化建议 (前8条):")
            for idx, rec in enumerate(analysis["recommendations"][:8], 1):
                print(f"   {idx}. {rec}")
        
        # 性能瓶颈
        bottlenecks = analysis["detailed_analysis"].get("performance_bottlenecks", {})
        if bottlenecks.get("slowest_cases"):
            print(f"\n🐌 最慢的查询用例 (前3个):")
            for idx, case in enumerate(bottlenecks["slowest_cases"][:3], 1):
                print(f"   {idx}. {case['test_id']}
"""资源管理器（确保资源释放）"""
    try:
        yield
    finally:
        if query_engine:
            try:
                query_engine.close()
                logger.info("✅ 查询引擎资源已释放")
            except Exception as e:
                logger.warning(f"⚠️ 关闭查询引擎时出错: {e}")

def main():
    """优化后的主函数"""
    parser = argparse.ArgumentParser(
        description="🚀 专业版向量数据库测试评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    示例用法:
    python test_evaluator.py --test_data test_set.json --db_path ./chroma_db
    python test_evaluator.py --test_data test_set.json --max_cases 100 --top_k 30 --no_progress
    python test_evaluator.py --test_data test_set.json --metrics enhanced --timeout 60
        """
    )
    
    parser.add_argument("--test_data", type=str, required=True, help="测试集JSON文件路径")
    parser.add_argument("--db_path", type=str, default="/tmp/chroma_db_dsw", help="ChromaDB数据库路径")
    parser.add_argument("--model_path", type=str, help="Embedding模型路径")
    parser.add_argument("--collection_name", type=str, default="rag_knowledge_base", help="集合名称")
    parser.add_argument("--output_dir", type=str, default="optimized_test_results", help="输出目录")
    parser.add_argument("--max_cases", type=int, default=None, help="最大测试用例数")
    parser.add_argument("--top_k", type=int, default=20, help="检索返回的Top-K数量")
    parser.add_argument("--metrics", type=str, choices=["basic", "enhanced"], default="enhanced", help="指标计算器类型")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有输出文件")
    parser.add_argument("--timeout", type=float, default=30.0, help="单用例超时时间(秒)")
    parser.add_argument("--retry", action="store_true", help="启用失败用例重试")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    parser.add_argument("--save_failed_only", action="store_true", help="仅保存失败用例详细结果")
    
    args = parser.parse_args()
    
    # 更新日志级别
    global logger
    logger = setup_logger(log_level=args.log_level)
    
    try:
        # 初始化评估器配置
        eval_config = {
            "max_test_cases": args.max_cases,
            "top_k": args.top_k,
            "enable_progress_bar": not args.no_progress,
            "overwrite_output": args.overwrite,
            "metrics_calculator": args.metrics,
            "timeout_per_case": args.timeout,
            "retry_failed_cases": args.retry,
            "save_failed_cases_only": args.save_failed_only
        }
        
        # 创建评估器
        logger.info("🛠️ 初始化测试评估器...")
        evaluator = OptimizedTestEvaluator(
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            config=eval_config
        )
        
        # 初始化查询引擎
        logger.info("🚀 初始化向量查询引擎...")
        query_args = {"db_path": args.db_path}
        if args.model_path:
            query_args["model_path"] = args.model_path
        if args.collection_name:
            query_args["collection_name"] = args.collection_name
        
        query_engine = OptimizedVectorDBQuery(**query_args)
        
        # 使用资源管理器
        with resource_manager(query_engine):
            # 测试连接
            logger.info("🔗 测试连接...")
            conn_test = query_engine.test_connection()
            
            # 格式化输出连接测试结果
            status_emoji = {
                True: "✅",
                False: "❌"
            }
            
            print("\n🔗 连接测试结果:")
            print("  " + "="*50)
            for key, value in conn_test.items():
                if key == "error_msg" and value:
                    print(f"    {key}: {value}")
                elif key != "error_msg":
                    # 处理字典类型的值（特别是term_dicts_loaded）
                    if isinstance(value, dict):
                        print(f"    {key}:")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, bool):
                                emoji = status_emoji.get(sub_value, "➖")
                                print(f"      {emoji} {sub_key}: {sub_value}")
                            else:
                                print(f"      {sub_key}: {sub_value}")
                    elif isinstance(value, bool):
                        emoji = status_emoji.get(value, "➖")
                        print(f"    {emoji} {key}: {value}")
                    else:
                        print(f"    {key}: {value}")
            
            # 检查核心连接状态
            if not conn_test.get("model_loaded"):
                raise RuntimeError("❌ 查询引擎初始化失败：Embedding模型加载异常")
            if not conn_test.get("db_connected"):
                raise RuntimeError("❌ 查询引擎初始化失败：ChromaDB数据库连接异常")
            if not conn_test.get("collection_exists"):
                logger.warning("⚠️ 警告：指定的集合不存在，将创建新集合")
            
            if conn_test.get("test_query_success"):
                logger.info("✅ 连接测试全部通过！")
            else:
                logger.warning("⚠️ 连接测试有警告，但继续执行...")
            
            print("  " + "="*50)
            
            # 运行测试
            logger.info("🧪 开始执行测试用例...")
            results = evaluator.run_test(query_engine)
            
            # 保存结果
            logger.info("💾 保存测试结果...")
            stats = evaluator.save_results(results, "vector_search")
            
            # 生成分析报告
            logger.info("📊 生成分析报告...")
            analysis = evaluator.generate_analysis(results, stats)
            
            # 打印总结
            evaluator.print_summary(stats, analysis)
            
            logger.info("🎉 测试评估流程全部完成!")
            
    except KeyboardInterrupt:
        logger.info("🛑 用户中断测试评估")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"💥 测试评估过程中发生致命错误: {e}")
        logger.error(f"详细错误:\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()