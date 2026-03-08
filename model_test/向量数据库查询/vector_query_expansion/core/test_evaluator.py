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
import warnings
import signal
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    List, Dict, Any, Optional, Union, Callable, Iterator, Tuple, Set
)
from collections import defaultdict, OrderedDict
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent          # core 目录
project_root = current_dir.parent                     # vector_query 目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入新版查询引擎（优化版）
try:
    from vector_query_optimized_new import OptimizedVectorDBQuery, QueryResult
except ImportError:
    try:
        from vector_db_query_optimized import OptimizedVectorDBQuery, QueryResult
    except ImportError:
        raise ImportError(
            "无法导入 OptimizedVectorDBQuery，请确保 vector_query_optimized_new.py 在 Python 路径中"
        )

# [新增] 导入查询扩展模块（可选）
try:
    from expansion import (
        BaseQueryExpander,
        SynonymExpander,
        PseudoRelevanceFeedbackExpander,
        ComposedExpander,
        load_expander_from_config,
    )
    EXPANSION_AVAILABLE = True
except ImportError:
    EXPANSION_AVAILABLE = False
    # 定义占位类型，避免语法错误
    class BaseQueryExpander: pass
    class SynonymExpander: pass
    class PseudoRelevanceFeedbackExpander: pass
    class ComposedExpander: pass
    def load_expander_from_config(*args, **kwargs): return None

# 
# 配置日志（支持结构化输出）
# 
def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    name: str = __name__
) -> logging.Logger:
    """初始化日志记录器（控制台+文件）"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter('%(asctime)s
"""评估器配置（推荐使用此配置类）"""
    # 测试数据配置
    test_data_path: str
    output_dir: str = "optimized_test_results"

    # 检索配置
    top_k: int = 20
    min_similarity: Optional[float] = None

    # 测试用例选择
    max_test_cases: Optional[int] = None
    query_type_filter: Optional[List[str]] = None
    difficulty_filter: Optional[List[str]] = None
    test_ids: Optional[List[str]] = None

    # 指标计算配置
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 15])
    metrics_calculator: str = "enhanced"

    # 行为控制
    enable_progress_bar: bool = True
    overwrite_output: bool = False
    save_detailed_results: bool = True
    save_failed_cases_only: bool = False
    save_csv: bool = True
    save_markdown: bool = False
    generate_charts: bool = False

    # 超时与重试
    timeout_per_case: float = 30.0
    retry_failed_cases: bool = False
    max_retries: int = 2
    batch_size: int = 10

    # 数据标准化
    normalize_doc_id_lowercase: bool = True

    # 统计扩展
    calculate_similarity_stats: bool = True
    analyze_retrieval_methods: bool = True
    analyze_cache_effect: bool = True

    # 断点续测
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 50

    # 日志
    log_dir: str = "logs"
    log_level: str = "INFO"

    # [新增] 查询扩展配置
    expander_config: Dict[str, Any] = field(default_factory=dict)
    """扩展器配置字典，例如：
    {
        "type": "synonym",
        "synonym_path": "./data/synonyms.txt",
        "max_synonyms": 2
    }
    或
    {
        "type": "prf",
        "top_k": 3,
        "top_terms": 5,
        "use_tfidf": true
    }
    """

    def __post_init__(self):
        """校验配置合法性"""
        if not self.test_data_path:
            raise ValueError("test_data_path 不能为空")
        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        for k in self.k_values:
            if k <= 0:
                raise ValueError("k_values 中的所有值必须为正整数")
        if self.timeout_per_case <= 0:
            raise ValueError("timeout_per_case 必须大于 0")
        if self.metrics_calculator not in ("basic", "enhanced"):
            raise ValueError("metrics_calculator 必须是 'basic' 或 'enhanced'")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluatorConfig":
        """从字典创建配置，忽略未知字段"""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        # 单独处理 expander_config（可能嵌套）
        if "expander_config" in config_dict:
            filtered["expander_config"] = config_dict["expander_config"]
        return cls(**filtered)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EvaluatorConfig":
        """从命令行参数解析配置"""
        config_dict = vars(args)
        return cls.from_dict(config_dict)


# 
# 指标计算器基类与实现（插件化）
# 
class MetricsCalculator(ABC):
    """指标计算器抽象基类"""

    def __init__(self, k_values: List[int]):
        self.k_values = sorted(k_values)

    @abstractmethod
    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        """计算指标，返回指标字典"""
        pass


class BasicMetricsCalculator(MetricsCalculator):
    """基础指标计算器（Hit@k, Precision@k, Recall@k, MRR, Recall）"""

    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        exp_set = set(filter(None, expected_ids))
        act_list = list(filter(None, actual_ids))

        if not exp_set:
            return {
                "is_valid_case": False,
                "error_type": "no_expected_docs",
                "note": "无预期文档，跳过指标计算"
            }

        if not act_list:
            return {
                "is_valid_case": True,
                "hit_at_k": {k: False for k in self.k_values},
                "precision_at_k": {k: 0.0 for k in self.k_values},
                "recall_at_k": {k: 0.0 for k in self.k_values},
                "mrr": 0.0,
                "recall": 0.0,
                "first_hit_rank": None,
                "hit_count": 0,
                "total_expected": len(exp_set)
            }

        hit_positions = []
        for idx, doc_id in enumerate(act_list, 1):
            if doc_id in exp_set:
                hit_positions.append(idx)

        hit_count = len(hit_positions)
        first_hit_rank = hit_positions[0] if hit_positions else None
        mrr = 1.0 / hit_positions[0] if hit_positions else 0.0

        hit_at_k = {}
        precision_at_k = {}
        recall_at_k = {}

        for k in self.k_values:
            hits_in_top_k = sum(1 for r in hit_positions if r <= k)
            hit_at_k[k] = hits_in_top_k > 0
            precision_at_k[k] = hits_in_top_k / min(k, len(act_list))
            recall_at_k[k] = hits_in_top_k / len(exp_set)

        recall = hit_count / len(exp_set)

        return {
            "is_valid_case": True,
            "hit_at_k": hit_at_k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "recall": recall,
            "first_hit_rank": first_hit_rank,
            "hit_count": hit_count,
            "total_expected": len(exp_set),
            "total_returned": len(act_list)
        }


class EnhancedMetricsCalculator(BasicMetricsCalculator):
    """增强指标计算器（增加 F1, AP, NDCG@k）"""

    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        base = super().calculate(expected_ids, actual_ids)
        if not base.get("is_valid_case", False):
            return base

        p10 = base["precision_at_k"].get(10, 0.0)
        r10 = base["recall_at_k"].get(10, 0.0)
        f1 = 2 * p10 * r10 / (p10 + r10) if (p10 + r10) > 0 else 0.0

        ap = self._average_precision(
            set(filter(None, expected_ids)),
            list(filter(None, actual_ids))
        )
        ndcg = self._ndcg_at_k(
            set(filter(None, expected_ids)),
            list(filter(None, actual_ids)),
            k=10
        )

        base.update({
            "f1_score": f1,
            "average_precision": ap,
            "ndcg_at_10": ndcg,
        })
        return base

    @staticmethod
    def _average_precision(exp_set: Set[str], act_list: List[str]) -> float:
        if not exp_set or not act_list:
            return 0.0
        relevant_count = 0
        prec_sum = 0.0
        for i, doc_id in enumerate(act_list, 1):
            if doc_id in exp_set:
                relevant_count += 1
                prec_sum += relevant_count / i
        return prec_sum / min(len(exp_set), len(act_list))

    @staticmethod
    def _ndcg_at_k(exp_set: Set[str], act_list: List[str], k: int) -> float:
        act_topk = act_list[:k]
        dcg = 0.0
        for i, doc_id in enumerate(act_topk, 1):
            if doc_id in exp_set:
                dcg += 1.0 / np.log2(i + 1)
        ideal_count = min(len(exp_set), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_count + 1))
        return dcg / idcg if idcg > 0 else 0.0


def create_metrics_calculator(
    calc_type: str,
    k_values: List[int]
) -> MetricsCalculator:
    if calc_type == "basic":
        return BasicMetricsCalculator(k_values)
    elif calc_type == "enhanced":
        return EnhancedMetricsCalculator(k_values)
    else:
        raise ValueError(f"未知的指标计算器类型: {calc_type}")


# 
# 测试结果数据类（增强字段，与新版查询引擎对齐）
# 
@dataclass
class TestResult:
    """单个测试用例的执行结果（
"""对测试结果进行多维统计与分析（性能优化版）"""def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.k_values = config.k_values

    def compute_summary_stats(self, results: List[TestResult]) -> Dict[str, Any]:
        total = len(results)
        success = [r for r in results if not r.is_failed]
        failed = [r for r in results if r.is_failed]
        valid = [
            r for r in success
            if r.metrics.get("is_valid_case", False)
        ]

        stats = {
            "summary": {
                "total_test_cases": total,
                "successful_cases": len(success),
                "failed_cases": len(failed),
                "valid_metric_cases": len(valid),
                "success_rate": len(success) / total if total else 0.0,
                "fail_rate": len(failed) / total if total else 0.0,
                "valid_case_rate": len(valid) / total if total else 0.0,
            }
        }

        if not valid:
            return stats

        # 性能指标
        times = np.array([r.retrieval_time for r in valid])
        perf = {
            "total_retrieval_time": float(np.sum(times)),
            "avg_retrieval_time": float(np.mean(times)),
            "retrieval_time_p50": float(np.percentile(times, 50)),
            "retrieval_time_p95": float(np.percentile(times, 95)),
            "retrieval_time_p99": float(np.percentile(times, 99)),
            "empty_result_count": sum(1 for r in success if not r.actual_doc_ids),
            "empty_result_rate": sum(1 for r in success if not r.actual_doc_ids) / len(success) if success else 0.0,
            "no_hit_count": sum(1 for r in valid if r.metrics.get("hit_count", 0) == 0),
            "no_hit_rate": sum(1 for r in valid if r.metrics.get("hit_count", 0) == 0) / len(valid),
        }

        for k in self.k_values:
            hit_arr = np.array([
                1 if r.metrics.get("hit_at_k", {}).get(k, False) else 0
                for r in valid
            ])
            prec_arr = np.array([
                r.metrics.get("precision_at_k", {}).get(k, 0.0)
                for r in valid
            ])
            rec_arr = np.array([
                r.metrics.get("recall_at_k", {}).get(k, 0.0)
                for r in valid
            ])
            perf[f"hit_at_{k}"] = float(np.mean(hit_arr))
            perf[f"precision_at_{k}"] = float(np.mean(prec_arr))
            perf[f"recall_at_{k}"] = float(np.mean(rec_arr))

        mrr_arr = np.array([r.metrics.get("mrr", 0.0) for r in valid])
        recall_arr = np.array([r.metrics.get("recall", 0.0) for r in valid])
        f1_arr = np.array([r.metrics.get("f1_score", 0.0) for r in valid])
        ap_arr = np.array([r.metrics.get("average_precision", 0.0) for r in valid])
        ndcg_arr = np.array([r.metrics.get("ndcg_at_10", 0.0) for r in valid])

        perf.update({
            "avg_mrr": float(np.mean(mrr_arr)),
            "avg_recall": float(np.mean(recall_arr)),
            "avg_f1_score": float(np.mean(f1_arr)),
            "avg_average_precision": float(np.mean(ap_arr)),
            "avg_ndcg_at_10": float(np.mean(ndcg_arr)),
        })

        first_hit_ranks = [
            r.metrics.get("first_hit_rank", 0)
            for r in valid if r.metrics.get("first_hit_rank") is not None
        ]
        if first_hit_ranks:
            perf["avg_first_hit_rank"] = float(np.mean(first_hit_ranks))
            perf["median_first_hit_rank"] = float(np.median(first_hit_ranks))
        else:
            perf["avg_first_hit_rank"] = 0.0
            perf["median_first_hit_rank"] = 0.0

        if self.config.calculate_similarity_stats:
            sim_means = []
            for r in success:
                if r.metadata.get("similarity_stats", {}).get("mean"):
                    sim_means.append(r.metadata["similarity_stats"]["mean"])
            if sim_means:
                sim_arr = np.array(sim_means)
                perf.update({
                    "avg_similarity": float(np.mean(sim_arr)),
                    "similarity_p25": float(np.percentile(sim_arr, 25)),
                    "similarity_p75": float(np.percentile(sim_arr, 75)),
                    "similarity_std": float(np.std(sim_arr)),
                })

        stats["performance"] = perf
        return stats

    def analyze_by_query_type(self, results: List[TestResult]) -> Dict[str, Any]:
        by_type = defaultdict(list)
        for r in results:
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                by_type[r.query_type].append(r)

        output = {}
        for qtype, cases in by_type.items():
            hit_arr = np.array([
                1 if c.metrics.get("hit_at_k", {}).get(1, False) else 0
                for c in cases
            ])
            time_arr = np.array([c.retrieval_time for c in cases])
            mrr_arr = np.array([c.metrics.get("mrr", 0.0) for c in cases])
            recall_arr = np.array([c.metrics.get("recall", 0.0) for c in cases])
            expansion_ratio = np.mean([1 if c.query_expansion_used else 0 for c in cases])

            output[qtype] = {
                "count": len(cases),
                "hit_at_1": float(np.mean(hit_arr)),
                "avg_time": float(np.mean(time_arr)),
                "avg_mrr": float(np.mean(mrr_arr)),
                "avg_recall": float(np.mean(recall_arr)),
                "expansion_ratio": float(expansion_ratio),  # [新增]
            }
        return output

    def analyze_by_retrieval_method(self, results: List[TestResult]) -> Dict[str, Any]:
        by_method = defaultdict(list)
        for r in results:
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                by_method[r.retrieval_method].append(r)

        output = {}
        for method, cases in by_method.items():
            hit_arr = np.array([
                1 if c.metrics.get("hit_at_k", {}).get(1, False) else 0
                for c in cases
            ])
            time_arr = np.array([c.retrieval_time for c in cases])
            mrr_arr = np.array([c.metrics.get("mrr", 0.0) for c in cases])
            reranked_ratio = np.mean([1 if c.reranked else 0 for c in cases])
            bm25_ratio = np.mean([1 if c.bm25_used else 0 for c in cases])
            expansion_ratio = np.mean([1 if c.query_expansion_used else 0 for c in cases])  # [新增]

            output[method] = {
                "count": len(cases),
                "hit_at_1": float(np.mean(hit_arr)),
                "avg_time": float(np.mean(time_arr)),
                "avg_mrr": float(np.mean(mrr_arr)),
                "reranked_ratio": float(reranked_ratio),
                "bm25_used_ratio": float(bm25_ratio),
                "expansion_ratio": float(expansion_ratio),  # [新增]
            }
        return output

    # [新增] 按是否使用查询扩展分析
    def analyze_by_expansion(self, results: List[TestResult]) -> Dict[str, Any]:
        """比较启用扩展与未启用扩展的测试用例表现"""
        expanded = []
        not_expanded = []
        for r in results:
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                if r.query_expansion_used:
                    expanded.append(r)
                else:
                    not_expanded.append(r)

        def _stats(group):
            if not group:
                return None
            hit1 = np.mean([1 if c.metrics.get("hit_at_k", {}).get(1, False) else 0 for c in group])
            mrr = np.mean([c.metrics.get("mrr", 0.0) for c in group])
            recall = np.mean([c.metrics.get("recall", 0.0) for c in group])
            time_avg = np.mean([c.retrieval_time for c in group])
            return {
                "count": len(group),
                "hit_at_1": float(hit1),
                "avg_mrr": float(mrr),
                "avg_recall": float(recall),
                "avg_time": float(time_avg),
            }

        result = {
            "expanded": _stats(expanded),
            "not_expanded": _stats(not_expanded),
        }

        # 计算提升率
        if result["expanded"] and result["not_expanded"]:
            exp = result["expanded"]
            base = result["not_expanded"]
            result["improvement"] = {
                "hit_at_1": (exp["hit_at_1"]
"""保存测试结果到多种格式"""def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, base: str, ext: str) -> str:
        if self.config.overwrite_output:
            return str(self.output_dir / f"{base}.{ext}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self.output_dir / f"{base}_{timestamp}.{ext}")

    def save_json(self, data: Dict[str, Any], base_name: str) -> str:
        path = self._safe_filename(base_name, "json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 已保存: {path}")
        return path

    def save_csv(self, results: List[TestResult], base_name: str) -> str:
        rows = []
        for r in results:
            row = {
                "test_id": r.test_id,
                "query_type": r.query_type,
                "difficulty": r.difficulty,
                "retrieval_method": r.retrieval_method,
                "reranked": r.reranked,
                "bm25_used": r.bm25_used,
                "query_expansion_used": r.query_expansion_used,  # [新增]
                "retrieval_time": r.retrieval_time,
                "is_failed": r.is_failed,
            }
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                row.update({
                    "hit_at_1": r.metrics.get("hit_at_k", {}).get(1, False),
                    "hit_at_3": r.metrics.get("hit_at_k", {}).get(3, False),
                    "hit_at_5": r.metrics.get("hit_at_k", {}).get(5, False),
                    "hit_at_10": r.metrics.get("hit_at_k", {}).get(10, False),
                    "mrr": r.metrics.get("mrr", 0.0),
                    "recall": r.metrics.get("recall", 0.0),
                    "f1_score": r.metrics.get("f1_score", 0.0),
                    "ndcg_at_10": r.metrics.get("ndcg_at_10", 0.0),
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        path = self._safe_filename(base_name, "csv")
        df.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"CSV 已保存: {path}")
        return path

    def save_markdown_report(
        self,
        stats: Dict[str, Any],
        analysis: Dict[str, Any],
        base_name: str
    ) -> str:
        lines = []
        lines.append("# 向量检索测试评估报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        s = stats["summary"]
        lines.append("## 1. 测试概览\n")
        lines.append(f"
"""优化版测试评估器 - 主控类"""def __init__(
        self,
        config: Union[EvaluatorConfig, Dict[str, Any]],
        query_engine: Optional[OptimizedVectorDBQuery] = None
    ):
        if isinstance(config, dict):
            self.config = EvaluatorConfig.from_dict(config)
        else:
            self.config = config

        self.data_loader = TestDataLoader(self.config)
        self.calculator = create_metrics_calculator(
            self.config.metrics_calculator,
            self.config.k_values
        )
        self.query_engine = query_engine
        self.executor: Optional[TestExecutor] = None
        self.analyzer = ResultAnalyzer(self.config)
        self.saver = ResultSaver(self.config)
        self.results: List[TestResult] = []

        self.checkpoint_file = None
        if self.config.resume_from_checkpoint:
            self._init_checkpoint()

    def _init_checkpoint(self):
        cp_dir = Path(self.config.output_dir) / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = cp_dir / f"checkpoint_{timestamp}.json"
        logger.info(f"断点文件将保存至: {self.checkpoint_file}")

    def set_query_engine(self, engine: OptimizedVectorDBQuery) -> None:
        self.query_engine = engine
        self.executor = TestExecutor(
            self.config, engine, self.calculator
        )

    def run(self) -> List[TestResult]:
        if self.query_engine is None:
            raise RuntimeError("未设置查询引擎，请调用 set_query_engine()")

        test_cases = self.data_loader.get_cases()
        total = len(test_cases)

        logger.info(f"开始执行 {total} 个测试用例...")
        self.results = []

        pbar = tqdm(
            total=total,
            desc="测试进度",
            unit="case",
            disable=not self.config.enable_progress_bar
        )

        for idx, case in enumerate(test_cases, 1):
            result = self.executor.execute_case(case)
            self.results.append(result)

            if self.config.enable_progress_bar:
                success_cnt = len([r for r in self.results if not r.is_failed])
                fail_cnt = len([r for r in self.results if r.is_failed])
                avg_time = np.mean([r.retrieval_time for r in self.results if not r.is_failed]) \
                    if success_cnt > 0 else 0
                pbar.set_postfix(
                    success=success_cnt,
                    fail=fail_cnt,
                    avg_time=f"{avg_time:.3f}s"
                )
                pbar.update(1)

            if self.checkpoint_file and idx % self.config.checkpoint_interval == 0:
                self._save_checkpoint(idx)

        pbar.close()
        logger.info(f"测试完成，成功 {len([r for r in self.results if not r.is_failed])} 例，"
                    f"失败 {len([r for r in self.results if r.is_failed])} 例")
        return self.results

    def _save_checkpoint(self, processed: int):
        if not self.checkpoint_file:
            return
        data = {
            "timestamp": datetime.now().isoformat(),
            "processed": processed,
            "total": len(self.data_loader.get_cases()),
            "results": [r.to_dict() for r in self.results]
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"检查点已保存，进度 {processed}/{len(self.data_loader.get_cases())}")

    def analyze(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("尚未运行测试，请先调用 run()")

        analysis = {
            "by_query_type": self.analyzer.analyze_by_query_type(self.results),
            "by_retrieval_method": self.analyzer.analyze_by_retrieval_method(self.results),
            "by_expansion": self.analyzer.analyze_by_expansion(self.results),  # [新增]
            "failed_cases_analysis": self.analyzer.analyze_failed_cases(self.results),
        }

        stats = self.analyzer.compute_summary_stats(self.results)
        analysis["stats"] = stats

        issues = []
        perf = stats.get("performance", {})
        if perf.get("hit_at_1", 0) < 0.3:
            issues.append(f"⚠️ Hit@1 = {perf['hit_at_1']*100:.1f}%，低于 30%")
        if perf.get("avg_retrieval_time", 0) > 1.0:
            issues.append(f"⚠️ 平均耗时 {perf['avg_retrieval_time']:.2f}s > 1s")
        if perf.get("empty_result_rate", 0) > 0.15:
            issues.append(f"⚠️ 空结果率 {perf['empty_result_rate']*100:.1f}% > 15%")
        if stats["summary"]["fail_rate"] > 0.1:
            issues.append(f"❌ 失败率 {stats['summary']['fail_rate']*100:.1f}% > 10%")
        analysis["issues"] = issues

        analysis["recommendations"] = self.analyzer.generate_recommendations(stats)

        return analysis

    def save_results(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        base = "vector_search"
        stats = analysis["stats"]

        saved["stats_json"] = self.saver.save_json(stats, f"{base}_stats")

        if self.config.save_detailed_results:
            results_to_save = self.results
            if self.config.save_failed_cases_only:
                results_to_save = [r for r in self.results if r.is_failed]
            if results_to_save:
                detailed_data = {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "config": asdict(self.config),
                        "total_cases": len(self.results),
                        "saved_cases": len(results_to_save)
                    },
                    "results": [r.to_dict() for r in results_to_save]
                }
                saved["detailed_json"] = self.saver.save_json(
                    detailed_data, f"{base}_detailed"
                )

        if self.config.save_csv:
            saved["csv"] = self.saver.save_csv(self.results, f"{base}_summary")

        if self.config.save_markdown:
            saved["markdown"] = self.saver.save_markdown_report(
                stats, analysis, f"{base}_report"
            )

        if self.config.generate_charts:
            self.saver.maybe_generate_charts(self.results)

        return saved

    def print_summary(self, analysis: Dict[str, Any]) -> None:
        stats = analysis["stats"]
        perf = stats.get("performance", {})
        s = stats["summary"]

        print("\n" + "=" * 100)
        print("📊 向量检索测试评估总结")
        print("=" * 100)

        print(f"\n🔍 测试概览:")
        print(f"   总用例: {s['total_test_cases']}, 成功: {s['successful_cases']} ({s['success_rate']*100:.1f}%), "
              f"失败: {s['failed_cases']} ({s['fail_rate']*100:.1f}%), 有效指标: {s['valid_metric_cases']}")

        print(f"\n⚡ 性能:")
        print(f"   平均耗时: {perf.get('avg_retrieval_time', 0):.3f}s, P95: {perf.get('retrieval_time_p95', 0):.3f}s")
        print(f"   空结果率: {perf.get('empty_result_rate', 0)*100:.1f}%, 无命中率: {perf.get('no_hit_rate', 0)*100:.1f}%")

        print(f"\n🎯 准确性:")
        for k in [1, 3, 5, 10]:
            print(f"   Hit@{k}: {perf.get(f'hit_at_{k}', 0)*100:5.1f}%")
        print(f"   MRR: {perf.get('avg_mrr', 0)*100:.1f}%, Recall: {perf.get('avg_recall', 0)*100:.1f}%, "
              f"NDCG@10: {perf.get('avg_ndcg_at_10', 0)*100:.1f}%")

        # [新增] 查询扩展对比
        exp_analysis = analysis.get("by_expansion", {})
        if exp_analysis.get("expanded") and exp_analysis.get("not_expanded"):
            print(f"\n🔁 查询扩展效果对比:")
            exp = exp_analysis["expanded"]
            base = exp_analysis["not_expanded"]
            imp = exp_analysis.get("improvement", {})
            print(f"   启用扩展: Hit@1={exp['hit_at_1']*100:.1f}% MRR={exp['avg_mrr']*100:.1f}% 耗时={exp['avg_time']:.3f}s")
            print(f"   未启用扩展: Hit@1={base['hit_at_1']*100:.1f}% MRR={base['avg_mrr']*100:.1f}% 耗时={base['avg_time']:.3f}s")
            print(f"   🔼 提升: Hit@1 {imp.get('hit_at_1',0)*100:+.1f}% MRR {imp.get('avg_mrr',0)*100:+.1f}% 耗时 {imp.get('avg_time',0)*100:+.1f}%")

        if analysis.get("issues"):
            print(f"\n⚠️ 发现问题:")
            for i, issue in enumerate(analysis["issues"][:5], 1):
                print(f"   {i}. {issue}")

        if analysis.get("recommendations"):
            print(f"\n💡 优化建议 (前5条):")
            for i, rec in enumerate(analysis["recommendations"][:5], 1):
                print(f"   {i}. {rec}")

        print("\n" + "=" * 100)
        print("✅ 评估完成，详细结果已保存至输出目录")
        print("=" * 100)


# 
# 资源管理器（上下文管理器）
"""安全创建并释放查询引擎，可选扩展器"""
    engine = None
    try:
        if model_path is None:
            model_path = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
        engine = OptimizedVectorDBQuery(
            db_path=db_path,
            model_path=model_path,
            collection_name=collection_name,
            expander=expander,  # [新增] 传递扩展器
            **kwargs
        )
        yield engine
    finally:
        if engine:
            try:
                engine.close()
                logger.info("查询引擎资源已释放")
            except Exception as e:
                logger.warning(f"释放查询引擎资源时出错: {e}")


# 
# [新增] 根据命令行参数构建扩展器
# 
def create_expander_from_args(args: argparse.Namespace) -> Optional[BaseQueryExpander]:
    """根据命令行参数创建查询扩展器实例"""
    if not EXPANSION_AVAILABLE:
        logger.warning("查询扩展模块未安装，无法创建扩展器")
        return None

    expander_type = args.expander
    if not expander_type or expander_type == "none":
        return None

    if expander_type == "synonym":
        return SynonymExpander(
            synonym_path=args.synonym_path or "./data/synonyms.txt",
            max_synonyms=args.max_synonyms or 2,
            expand_all_tokens=True,
        )
    elif expander_type == "prf":
        # PRF 需要检索器实例，此时无法创建，需要延迟创建或采用其他方式
        logger.error("PRF 扩展器需要检索器实例，无法在命令行直接构建。请通过配置文件或代码动态创建。")
        return None
    elif expander_type == "compose":
        logger.error("组合扩展器暂不支持命令行直接创建，请使用配置文件。")
        return None
    else:
        logger.error(f"未知扩展器类型: {expander_type}")
        return None


# 
# 命令行入口
# 
def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 向量检索测试评估工具 v2.1（支持查询扩展）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    使用示例:
    python test_evaluator.py --test_data test_set.json --db_path ./chroma_db
    python test_evaluator.py --test_data test_set.json --expander synonym --synonym_path ./data/synonyms.txt
        """
    )
    # 必须参数
    parser.add_argument("--test_data", type=str, required=True, help="测试集JSON文件路径")

    # 检索配置
    parser.add_argument("--db_path", type=str, default="/tmp/chroma_db_dsw", help="ChromaDB数据库路径")
    parser.add_argument("--model_path", type=str, help="Embedding模型路径（默认自动）")
    parser.add_argument("--collection_name", type=str, default="rag_knowledge_base", help="集合名称")
    parser.add_argument("--top_k", type=int, default=20, help="检索返回数量")
    parser.add_argument("--min_similarity", type=float, default=None, help="最小相似度阈值")

    # 测试用例过滤
    parser.add_argument("--max_cases", type=int, default=None, help="最大执行用例数")
    parser.add_argument("--query_type", type=str, nargs="+", help="按查询类型过滤")
    parser.add_argument("--difficulty", type=str, nargs="+", help="按难度过滤")
    parser.add_argument("--test_ids", type=str, nargs="+", help="指定执行ID列表")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="optimized_test_results", help="输出目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖输出文件")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--save_failed_only", action="store_true", help="仅保存失败用例详细结果")
    parser.add_argument("--no_csv", action="store_true", help="不保存CSV")
    parser.add_argument("--markdown", action="store_true", help="生成Markdown报告")
    parser.add_argument("--charts", action="store_true", help="生成图表（需matplotlib）")

    # 指标与性能
    parser.add_argument("--metrics", type=str, choices=["basic", "enhanced"], default="enhanced", help="指标计算器")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 10, 15], help="自定义k值列表")
    parser.add_argument("--timeout", type=float, default=30.0, help="单用例超时(秒)")
    parser.add_argument("--retry", action="store_true", help="失败重试")
    parser.add_argument("--max_retries", type=int, default=2, help="最大重试次数")
    parser.add_argument("--batch_size", type=int, default=10, help="进度更新批次大小")

    # [新增] 查询扩展参数
    parser.add_argument("--expander", type=str, choices=["none", "synonym", "prf", "compose"], default="none",
                        help="查询扩展器类型")
    parser.add_argument("--synonym_path", type=str, default="./data/synonyms.txt", help="同义词词典路径")
    parser.add_argument("--max_synonyms", type=int, default=2, help="最大同义词数量")
    # PRF 参数（暂不支持命令行）
    parser.add_argument("--prf_top_k", type=int, default=3, help="PRF首次检索文档数（需代码集成）")
    parser.add_argument("--prf_top_terms", type=int, default=5, help="PRF提取关键词数（需代码集成）")

    # 其他
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="启用断点续测（实验性）")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="检查点保存间隔")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    global logger
    logger = setup_logger(log_level=args.log_level)

    try:
        # 1. 构建配置对象
        config_dict = {
            "test_data_path": args.test_data,
            "output_dir": args.output_dir,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "max_test_cases": args.max_cases,
            "query_type_filter": args.query_type,
            "difficulty_filter": args.difficulty,
            "test_ids": args.test_ids,
            "k_values": args.k_values,
            "metrics_calculator": args.metrics,
            "enable_progress_bar": not args.no_progress,
            "overwrite_output": args.overwrite,
            "save_detailed_results": True,
            "save_failed_cases_only": args.save_failed_only,
            "save_csv": not args.no_csv,
            "save_markdown": args.markdown,
            "generate_charts": args.charts,
            "timeout_per_case": args.timeout,
            "retry_failed_cases": args.retry,
            "max_retries": args.max_retries,
            "batch_size": args.batch_size,
            "resume_from_checkpoint": args.resume,
            "checkpoint_interval": args.checkpoint_interval,
            "log_level": args.log_level,
            # [新增] 扩展器配置（存入字典）
            "expander_config": {
                "type": args.expander,
                "synonym_path": args.synonym_path,
                "max_synonyms": args.max_synonyms,
                "prf_top_k": args.prf_top_k,
                "prf_top_terms": args.prf_top_terms,
            }
        }

        config = EvaluatorConfig.from_dict(config_dict)

        # 2. 初始化评估器
        evaluator = OptimizedTestEvaluator(config)

        # 3. 创建查询扩展器（仅支持同义词，PRF需动态创建）
        expander = None
        if args.expander == "synonym" and EXPANSION_AVAILABLE:
            expander = SynonymExpander(
                synonym_path=args.synonym_path,
                max_synonyms=args.max_synonyms,
            )
            logger.info(f"已创建同义词扩展器，词典: {args.synonym_path}")
        elif args.expander != "none":
            logger.warning(f"扩展器类型 '{args.expander}' 暂不支持命令行直接创建，跳过")

        # 4. 创建查询引擎（传入扩展器）
        with managed_query_engine(
            db_path=args.db_path,
            model_path=args.model_path,
            collection_name=args.collection_name,
            expander=expander,
        ) as engine:
            logger.info("🔗 测试连接状态...")
            conn = engine.test_connection()
            if not conn.get("model_loaded"):
                raise RuntimeError("Embedding 模型加载失败")
            if not conn.get("db_connected"):
                raise RuntimeError("ChromaDB 连接失败")
            logger.info("✅ 连接测试通过")

            evaluator.set_query_engine(engine)
            results = evaluator.run()

        # 5. 分析结果
        logger.info("📊 分析测试结果...")
        analysis = evaluator.analyze()

        # 6. 保存结果
        logger.info("💾 保存结果...")
        evaluator.save_results(analysis)

        # 7. 打印总结
        evaluator.print_summary(analysis)

    except KeyboardInterrupt:
        logger.info("🛑 用户中断")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"💥 执行失败: {e}\n{traceback.format_exc()}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()