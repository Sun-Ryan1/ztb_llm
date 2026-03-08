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
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入新版查询引擎（优化版）
try:
    from vector_query_optimized_new import OptimizedVectorDBQuery, QueryResult
except ImportError:
    # 兼容旧命名
    try:
        from vector_db_query_optimized import OptimizedVectorDBQuery, QueryResult
    except ImportError:
        raise ImportError(
            "无法导入 OptimizedVectorDBQuery，请确保 vector_query_optimized_new.py 在 Python 路径中"
        )

# 
# 配置日志（支持结构化输出）
# 
def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    name: str = __name__
) -> logging.Logger:
    """初始化日志记录器（控制台+文件）
"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger

    # 控制台 Handler
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter('%(asctime)s
"""评估器配置（推荐使用此配置类）
"""
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
    metrics_calculator: str = "enhanced"          # "basic" 或 "enhanced"

    # 行为控制
    enable_progress_bar: bool = True
    overwrite_output: bool = False
    save_detailed_results: bool = True
    save_failed_cases_only: bool = False
    save_csv: bool = True
    save_markdown: bool = False
    generate_charts: bool = False                 # 需要 matplotlib

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

    def __post_init__(self):
        """校验配置合法性
"""
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
        """从字典创建配置，忽略未知字段
"""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EvaluatorConfig":
        """从命令行参数解析配置
"""
        config_dict = vars(args)
        return cls.from_dict(config_dict)

# 
# 指标计算器基类与实现（插件化）
# 
class MetricsCalculator(ABC):
    """指标计算器抽象基类
"""

    def __init__(self, k_values: List[int]):
        self.k_values = sorted(k_values)

    @abstractmethod
    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        """计算指标，返回指标字典
"""
        pass

class BasicMetricsCalculator(MetricsCalculator):
    """基础指标计算器（Hit@k, Precision@k, Recall@k, MRR, Recall）
"""

    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        # 标准化ID（过滤空值）
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

        # 预先计算命中位置
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
    """增强指标计算器（增加 F1, AP, NDCG@k）
"""

    def calculate(
        self,
        expected_ids: List[str],
        actual_ids: List[str]
    ) -> Dict[str, Any]:
        base = super().calculate(expected_ids, actual_ids)
        if not base.get("is_valid_case", False):
            return base

        # 计算 F1@10
        p10 = base["precision_at_k"].get(10, 0.0)
        r10 = base["recall_at_k"].get(10, 0.0)
        f1 = 2 * p10 * r10 / (p10 + r10) if (p10 + r10) > 0 else 0.0

        # 计算 AP (Average Precision)
        ap = self._average_precision(
            set(filter(None, expected_ids)),
            list(filter(None, actual_ids))
        )

        # 计算 NDCG@10
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
    """指标计算器工厂函数
"""
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
"""标准化文档ID"""if doc_id is None:
            return ""
        doc_id = str(doc_id).strip()
        if self.config.normalize_doc_id_lowercase:
            doc_id = doc_id.lower()
        return doc_id

    @staticmethod
    def _calc_similarity_stats(docs: List[Dict]) -> Dict[str, float]:
"""计算返回文档的相似度统计"""sims = [d.get("similarity", 0.0) for d in docs if d.get("similarity") is not None]
        if not sims:
            return {}
        return {
            "min": float(np.min(sims)),
            "max": float(np.max(sims)),
            "mean": float(np.mean(sims)),
            "median": float(np.median(sims)),
            "std": float(np.std(sims)) if len(sims) > 1 else 0.0,
        }

    def _build_failed_result(
        self,
        test_case: Dict[str, Any],
        error_msg: str,
        is_timeout: bool = False
    ) -> TestResult:
"""构造失败结果"""return TestResult(
            test_id=test_case.get("test_id", "unknown"),
            query=test_case.get("query", ""),
            query_type=test_case.get("query_type", "general"),
            difficulty=test_case.get("difficulty", "medium"),
            expected_doc_ids=test_case.get("expected_doc_ids", []),
            actual_doc_ids=[],
            retrieval_method="failed",
            retrieval_time=0.0,
            reranked=False,
            bm25_used=False,
            top_k=self.config.top_k,
            metrics={"is_valid_case": False, "error_type": "timeout" if is_timeout else "exception"},
            retrieved_docs=[],
            is_failed=True,
            error_msg=error_msg,
            warning_msgs=[],
            metadata={"is_timeout": is_timeout, "retry_count": 0}
        )

# 
# 统计分析器（NumPy 向量化优化）
# 
class ResultAnalyzer:
"""对测试结果进行多维统计与分析（性能优化版）"""def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.k_values = config.k_values

    def compute_summary_stats(self, results: List[TestResult]) -> Dict[str, Any]:
"""计算核心统计指标（NumPy 向量化）"""total = len(results)
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

        # ----
"""按查询类型聚合"""by_type = defaultdict(list)
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

            output[qtype] = {
                "count": len(cases),
                "hit_at_1": float(np.mean(hit_arr)),
                "avg_time": float(np.mean(time_arr)),
                "avg_mrr": float(np.mean(mrr_arr)),
                "avg_recall": float(np.mean(recall_arr)),
            }
        return output

    def analyze_by_retrieval_method(self, results: List[TestResult]) -> Dict[str, Any]:
"""按检索方法聚合"""by_method = defaultdict(list)
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

            output[method] = {
                "count": len(cases),
                "hit_at_1": float(np.mean(hit_arr)),
                "avg_time": float(np.mean(time_arr)),
                "avg_mrr": float(np.mean(mrr_arr)),
                "reranked_ratio": float(reranked_ratio),
                "bm25_used_ratio": float(bm25_ratio),
            }
        return output

    def analyze_failed_cases(self, results: List[TestResult]) -> Dict[str, Any]:
"""失败用例详细分析"""failed = [r for r in results if r.is_failed]
        if not failed:
            return {"total_failed": 0}

        error_types = defaultdict(int)
        timeout_ids = []
        for r in failed:
            err = r.error_msg.split(":")[0] if ":" in r.error_msg else r.error_msg
            error_types[err] += 1
            if r.metadata.get("is_timeout", False):
                timeout_ids.append(r.test_id)

        # 按查询类型失败率
        qtype_fail = defaultdict(lambda: {"total": 0, "failed": 0})
        for r in results:
            qtype = r.query_type
            qtype_fail[qtype]["total"] += 1
            if r.is_failed:
                qtype_fail[qtype]["failed"] += 1

        fail_rate_by_qtype = {
            q: d["failed"] / d["total"] if d["total"] > 0 else 0.0
            for q, d in qtype_fail.items()
        }

        return {
            "total_failed": len(failed),
            "error_distribution": dict(error_types),
            "timeout_cases": timeout_ids[:10],
            "failure_rate_by_query_type": dict(sorted(
                fail_rate_by_qtype.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

    def generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
"""基于统计信息生成优化建议"""recs = []
        perf = stats.get("performance", {})

        # 命中率建议
        hit1 = perf.get("hit_at_1", 0.0)
        if hit1 < 0.2:
            recs.append("❌ 致命: Hit@1 < 20%，检索基本失效 → 检查 embedding 质量 / 文档内容对齐")
        elif hit1 < 0.4:
            recs.append("⚠️ 严重: Hit@1 偏低 → 尝试查询扩展、重排序或领域微调")
        elif hit1 < 0.6:
            recs.append("ℹ️ 一般: Hit@1 有提升空间 → 优化混合权重或增加候选集")

        # 性能建议
        avg_t = perf.get("avg_retrieval_time", 0.0)
        if avg_t > 2.0:
            recs.append("❌ 致命: 平均响应时间 > 2s → 必须优化：启用 BM25 全局索引、减少重排序候选集")
        elif avg_t > 1.0:
            recs.append("⚠️ 严重: 平均响应时间 > 1s → 建议使用向量缓存、升级硬件")
        elif avg_t > 0.5:
            recs.append("ℹ️ 一般: 平均响应时间 > 0.5s → 可考虑异步查询或量化模型")

        # 空结果率
        empty = perf.get("empty_result_rate", 0.0)
        if empty > 0.3:
            recs.append("❌ 致命: 空结果 > 30% → 检查数据覆盖或查询改写逻辑")
        elif empty > 0.1:
            recs.append("⚠️ 严重: 空结果 > 10% → 增加兜底策略，分析高频空查询")

        # 无命中率
        no_hit = perf.get("no_hit_rate", 0.0)
        if no_hit > 0.5:
            recs.append("❌ 致命: 超过 50% 的有效查询无任何命中文档 → 严重数据缺失或检索策略错误")
        elif no_hit > 0.3:
            recs.append("⚠️ 严重: 超过 30% 的有效查询无命中文档 → 需紧急优化")

        # 通用建议
        recs.extend([
            "📊 建立性能基线，每次优化后运行测试集",
            "🔧 考虑集成 CI/CD 自动评估",
            "🧪 增加压力测试，评估并发能力"
        ])

        return recs[:10]  # 最多10条

# 
# 结果保存器（支持 JSON / CSV / Markdown / 图表）
# 
class ResultSaver:
"""保存测试结果到多种格式"""def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, base: str, ext: str) -> str:
"""生成安全的文件名（避免覆盖或添加时间戳）"""if self.config.overwrite_output:
            return str(self.output_dir / f"{base}.{ext}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(self.output_dir / f"{base}_{timestamp}.{ext}")

    def save_json(self, data: Dict[str, Any], base_name: str) -> str:
"""保存为 JSON 文件"""path = self._safe_filename(base_name, "json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 已保存: {path}")
        return path

    def save_csv(self, results: List[TestResult], base_name: str) -> str:
"""保存结果摘要为 CSV"""rows = []
        for r in results:
            row = {
                "test_id": r.test_id,
                "query_type": r.query_type,
                "difficulty": r.difficulty,
                "retrieval_method": r.retrieval_method,
                "reranked": r.reranked,
                "bm25_used": r.bm25_used,
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
"""生成 Markdown 格式的报告"""lines = []
        lines.append("# 向量检索测试评估报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 概览
        s = stats["summary"]
        lines.append("## 1. 测试概览\n")
        lines.append(f"
"""生成可视化图表（可选，依赖 matplotlib）"""if not self.config.generate_charts:
            return
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("未安装 matplotlib，跳过图表生成")
            return

        # 检索时间分布
        times = [r.retrieval_time for r in results if not r.is_failed]
        if times:
            plt.figure(figsize=(10, 6))
            plt.hist(times, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel("检索时间 (s)")
            plt.ylabel("频次")
            plt.title("检索时间分布")
            plt.grid(True, linestyle='--', alpha=0.5)
            chart_path = self.output_dir / "retrieval_time_dist.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"图表已保存: {chart_path}")

        # Hit@1 按查询类型
        by_type = defaultdict(list)
        for r in results:
            if not r.is_failed and r.metrics.get("is_valid_case", False):
                by_type[r.query_type].append(r)
        if by_type:
            types = []
            hit1_vals = []
            for qtype, cases in by_type.items():
                hit1 = np.mean([1 if c.metrics.get("hit_at_k", {}).get(1, False) else 0 for c in cases])
                types.append(qtype)
                hit1_vals.append(hit1)
            plt.figure(figsize=(12, 6))
            plt.bar(types, hit1_vals)
            plt.xlabel("查询类型")
            plt.ylabel("Hit@1")
            plt.title("各查询类型 Hit@1 表现")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            chart_path = self.output_dir / "hit1_by_query_type.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"图表已保存: {chart_path}")

# 
# 主评估器（协调各组件）
# 
class OptimizedTestEvaluator:
"""优化版测试评估器
"""初始化断点续测
"""
        cp_dir = Path(self.config.output_dir) / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = cp_dir / f"checkpoint_{timestamp}.json"
        logger.info(f"断点文件将保存至: {self.checkpoint_file}")

    def set_query_engine(self, engine: OptimizedVectorDBQuery) -> None:
        """设置查询引擎（必须在 run 之前调用）
"""
        self.query_engine = engine
        self.executor = TestExecutor(
            self.config, engine, self.calculator
        )

    def run(self) -> List[TestResult]:
        """执行全部测试用例
"""
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

            # 定期保存检查点
            if self.checkpoint_file and idx % self.config.checkpoint_interval == 0:
                self._save_checkpoint(idx)

        pbar.close()
        logger.info(f"测试完成，成功 {len([r for r in self.results if not r.is_failed])} 例，"
                    f"失败 {len([r for r in self.results if r.is_failed])} 例")
        return self.results

    def _save_checkpoint(self, processed: int):
        """保存当前进度到检查点文件
"""
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
        """对结果进行多维分析
"""
        if not self.results:
            raise RuntimeError("尚未运行测试，请先调用 run()")

        analysis = {
            "by_query_type": self.analyzer.analyze_by_query_type(self.results),
            "by_retrieval_method": self.analyzer.analyze_by_retrieval_method(self.results),
            "failed_cases_analysis": self.analyzer.analyze_failed_cases(self.results),
        }

        # 汇总统计
        stats = self.analyzer.compute_summary_stats(self.results)
        analysis["stats"] = stats

        # 生成问题列表
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

        # 生成建议
        analysis["recommendations"] = self.analyzer.generate_recommendations(stats)

        return analysis

    def save_results(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """保存所有输出文件，返回文件路径映射
"""
        saved = {}
        base = "vector_search"
        stats = analysis["stats"]

        # 1. 统计信息 JSON
        saved["stats_json"] = self.saver.save_json(stats, f"{base}_stats")

        # 2. 详细结果 JSON（可选）
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

        # 3. CSV 摘要
        if self.config.save_csv:
            saved["csv"] = self.saver.save_csv(self.results, f"{base}_summary")

        # 4. Markdown 报告
        if self.config.save_markdown:
            saved["markdown"] = self.saver.save_markdown_report(
                stats, analysis, f"{base}_report"
            )

        # 5. 图表（可选）
        if self.config.generate_charts:
            self.saver.maybe_generate_charts(self.results)

        return saved

    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """控制台输出简洁总结
"""
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
# 
@contextmanager
def managed_query_engine(
    db_path: str = "/tmp/chroma_db_dsw",
    model_path: Optional[str] = None,
    collection_name: str = "rag_knowledge_base",
    **kwargs
) -> Iterator[OptimizedVectorDBQuery]:
    """安全创建并释放查询引擎
"""
    engine = None
    try:
        if model_path is None:
            # 使用默认路径（可根据环境调整）
            model_path = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
        engine = OptimizedVectorDBQuery(
            db_path=db_path,
            model_path=model_path,
            collection_name=collection_name,
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
# 命令行入口
# 
def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 向量检索测试评估工具 v2.0（专业优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
    python test_evaluator_optimized.py --test_data test_set.json --db_path ./chroma_db
    python test_evaluator_optimized.py --test_data test_set.json --max_cases 100 --top_k 30 --no_progress
    python test_evaluator_optimized.py --test_data test_set.json --metrics enhanced --timeout 60 --retry
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

    # 其他
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="启用断点续测（实验性）")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="检查点保存间隔")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 更新日志级别
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
        }

        config = EvaluatorConfig.from_dict(config_dict)

        # 2. 初始化评估器（此时尚未传入引擎）
        evaluator = OptimizedTestEvaluator(config)

        # 3. 创建查询引擎（使用资源管理器）
        with managed_query_engine(
            db_path=args.db_path,
            model_path=args.model_path,
            collection_name=args.collection_name
        ) as engine:
            # 连接测试
            logger.info("🔗 测试连接状态...")
            conn = engine.test_connection()
            if not conn.get("model_loaded"):
                raise RuntimeError("Embedding 模型加载失败")
            if not conn.get("db_connected"):
                raise RuntimeError("ChromaDB 连接失败")
            logger.info("✅ 连接测试通过")

            # 设置引擎并运行
            evaluator.set_query_engine(engine)
            results = evaluator.run()

        # 4. 分析结果
        logger.info("📊 分析测试结果...")
        analysis = evaluator.analyze()

        # 5. 保存结果
        logger.info("💾 保存结果...")
        evaluator.save_results(analysis)

        # 6. 打印总结
        evaluator.print_summary(analysis)

    except KeyboardInterrupt:
        logger.info("🛑 用户中断")
        raise SystemExit(0)
    except Exception as e:
        logger.error(f"💥 执行失败: {e}\n{traceback.format_exc()}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()