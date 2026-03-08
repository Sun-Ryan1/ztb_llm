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
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import contextmanager
from tqdm import tqdm

# 第三方依赖导入放在try-except中，增强兼容性
try:
    from vector_db_query_optimized import OptimizedVectorDBQuery, QueryResult
except ImportError as e:
    logging.error(f"导入向量查询模块失败: {e}")
    raise

# 配置日志（支持文件输出）
def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """初始化日志配置
"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复处理器
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s
"""测试结果数据类
"""test_id: str
    query: str
    expected_doc_ids: List[str]
    actual_doc_ids: List[str]
    query_type: str
    difficulty: str
    retrieval_method: str
    retrieval_time: float
    top_k: int
    hit_at_k: Dict[int, bool]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    retrieved_docs: List[Dict] = field(default_factory=list)
    is_failed: bool = False  # 新增：标记用例是否执行失败
    error_msg: str = ""      # 新增：失败时的错误信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容原有格式+新增失败信息）
"""
        base_dict = {
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
            # 各k值指标
            **{f"hit_at_{k}": self.hit_at_k.get(k, False) for k in [1,3,5,10,15]},
            **{f"precision_at_{k}": self.precision_at_k.get(k, 0.0) for k in [1,3,5,10,15]},
            **{f"recall_at_{k}": self.recall_at_k.get(k, 0.0) for k in [1,3,5,10,15]},
            "mrr": self.mrr
        }
        return base_dict

class OptimizedTestEvaluator:
    """优化版测试评估器
"""
    
    DEFAULT_CONFIG = {
        "max_test_cases": None,
        "k_values": [1, 3, 5, 10, 15],
        "enable_progress_bar": True,
        "top_k": 20,
        "normalize_doc_id_lowercase": True,  # 新增：可配置是否转小写
        "overwrite_output": False,           # 新增：是否覆盖输出文件
        "log_dir": "logs"                    # 新增：日志目录
    }
    
    def __init__(self, 
                 test_data_path: str, 
                 output_dir: str = "optimized_test_results",
                 config: Optional[Dict[str, Any]] = None):
        """初始化评估器
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
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        logger.info(f"加载测试数据完成: 共{len(self.test_data['test_cases'])}个测试用例")
    
    def _validate_config(self) -> None:
        """校验配置参数合法性
"""
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
        for bool_key in ["enable_progress_bar", "normalize_doc_id_lowercase", "overwrite_output"]:
            if not isinstance(self.config[bool_key], bool):
                raise ValueError(f"{bool_key}必须是布尔值")
    
    def load_test_data(self) -> Dict[str, Any]:
        """加载并校验测试数据
"""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 基础校验
            if "test_cases" not in data or not isinstance(data["test_cases"], list):
                raise ValueError("测试数据必须包含test_cases列表")
            
            # 校验每个用例的必填字段
            required_fields = ["test_id", "query", "expected_doc_ids"]
            for idx, case in enumerate(data["test_cases"]):
                missing_fields = [f for f in required_fields if f not in case]
                if missing_fields:
                    raise ValueError(f"测试用例{idx}缺少必填字段: {missing_fields}")
            
            return data
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}")
            raise
    
    def _normalize_doc_id(self, doc_id: Union[str, int, None]) -> str:
        """标准化文档ID（增强鲁棒性）
        :param doc_id: 原始文档ID（支持字符串/整数/None）
        :return: 标准化后的ID
"""
        if doc_id is None:
            return ""
        
        # 统一转为字符串
        if not isinstance(doc_id, str):
            doc_id = str(doc_id)
        
        # 基础处理
        doc_id = doc_id.strip().replace(" ", "")
        
        # 可选转小写
        if self.config["normalize_doc_id_lowercase"]:
            doc_id = doc_id.lower()
        
        return doc_id
    
    def calculate_metrics(self, expected_ids: List[str], actual_ids: List[str]) -> tuple:
        """计算核心指标（修复逻辑错误）
        :param expected_ids: 预期文档ID列表
        :param actual_ids: 实际返回文档ID列表
        :return: (hit_at_k, precision_at_k, recall_at_k, mrr)
"""
        hit_at_k = {k: False for k in self.config["k_values"]}
        precision_at_k = {k: 0.0 for k in self.config["k_values"]}
        recall_at_k = {k: 0.0 for k in self.config["k_values"]}
        
        # 标准化ID（过滤空值）
        normalized_expected = set([
            self._normalize_doc_id(id) for id in expected_ids if id
        ])
        normalized_actual = [
            self._normalize_doc_id(id) for id in actual_ids if id
        ]
        
        # 处理无预期文档的边界情况
        if not normalized_expected:
            # 无预期时：所有k值都算命中，精确率/召回率为1.0
            for k in self.config["k_values"]:
                hit_at_k[k] = True
                precision_at_k[k] = 1.0
                recall_at_k[k] = 1.0
            mrr = 1.0  # 无预期时MRR为1.0
            return hit_at_k, precision_at_k, recall_at_k, mrr
        
        # 处理有预期文档的情况
        for k in self.config["k_values"]:
            top_k_actual = normalized_actual[:k]
            # 计算命中数
            hits = sum(1 for doc_id in top_k_actual if doc_id in normalized_expected)
            
            # 计算各指标
            hit_at_k[k] = hits > 0
            precision_at_k[k] = hits / k if k > 0 else 0.0
            recall_at_k[k] = hits / len(normalized_expected) if normalized_expected else 0.0
        
        # 修复MRR计算逻辑（取最高排名的倒数）
        reciprocal_ranks = []
        for idx, doc_id in enumerate(normalized_actual):
            if doc_id in normalized_expected:
                reciprocal_ranks.append(1.0 / (idx + 1))
        mrr = max(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        return hit_at_k, precision_at_k, recall_at_k, mrr
    
    def _create_failed_test_result(self, test_case: Dict[str, Any], error_msg: str) -> TestResult:
        """创建失败用例的结果对象
"""
        empty_k_dict = {k: False for k in self.config["k_values"]}
        empty_float_dict = {k: 0.0 for k in self.config["k_values"]}
        
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
            hit_at_k=empty_k_dict,
            precision_at_k=empty_float_dict,
            recall_at_k=empty_float_dict,
            mrr=0.0,
            is_failed=True,
            error_msg=error_msg
        )
    
    @contextmanager
    def _resource_manager(self, query_engine: OptimizedVectorDBQuery):
        """资源管理器（确保资源释放）
"""
        try:
            yield
        finally:
            try:
                query_engine.close()
                logger.info("查询引擎资源已释放")
            except Exception as e:
                logger.warning(f"关闭查询引擎时出错: {e}")
    
    def run_test(self, query_engine: OptimizedVectorDBQuery) -> List[TestResult]:
        """运行测试（增强异常处理，记录失败用例）
        :param query_engine: 向量查询引擎实例
        :return: 所有测试结果（含失败用例）
"""
        test_cases = self.test_data["test_cases"]
        
        # 限制测试用例数量
        if self.config["max_test_cases"] is not None:
            test_cases = test_cases[:self.config["max_test_cases"]]
        
        total_cases = len(test_cases)
        logger.info(f"开始执行测试: 共{total_cases}个测试用例")
        
        results = []
        pbar = tqdm(test_cases, desc="测试进度") if self.config["enable_progress_bar"] else test_cases
        
        with self._resource_manager(query_engine):
            for test_case in pbar:
                try:
                    # 基础信息提取
                    test_id = test_case["test_id"]
                    query = test_case["query"]
                    expected_doc_ids = test_case["expected_doc_ids"]
                    query_type = test_case.get("query_type", "general")
                    difficulty = test_case.get("difficulty", "medium")
                    
                    # 执行查询
                    start_time = time.time()
                    query_result = query_engine.query(
                        query_text=query,
                        top_k=self.config["top_k"],
                        return_format="structured"
                    )
                    retrieval_time = time.time()
"""获取安全的文件路径（避免覆盖）
        :param filename: 基础文件名
        :param ext: 文件扩展名
        :return: 安全的文件路径
"""base_path = os.path.join(self.output_dir, f"{filename}.{ext}")
        
        # 如果允许覆盖，直接返回
        if self.config["overwrite_output"]:
            return base_path
        
        # 避免覆盖：添加数字后缀
        counter = 1
        safe_path = base_path
        while os.path.exists(safe_path):
            safe_path = os.path.join(self.output_dir, f"{filename}_{counter}.{ext}")
            counter += 1
        
        return safe_path
    
    def save_results(self, results: List[TestResult], filename: str = "test_results") -> Dict[str, Any]:
        """保存结果（优化文件覆盖、增强容错）
        :param results: 测试结果列表
        :param filename: 基础文件名
        :return: 统计信息
"""
        if not results:
            logger.warning("无测试结果可保存")
            return {}
        
        try:
            # 1. 保存详细结果
            detailed_path = self._get_safe_file_path(f"{filename}_detailed")
            simplified_results = []
            for r in results:
                simplified = r.to_dict()
                # 添加文档预览（增强信息）
                if r.retrieved_docs:
                    simplified["retrieved_docs_preview"] = [
                        {
                            "id": doc.get("id", ""),
                            "content_preview": doc.get("content_preview", "")[:200],  # 截断长内容
                            "similarity": doc.get("similarity", 0.0)
                        }
                        for doc in r.retrieved_docs[:3]
                    ]
                simplified_results.append(simplified)
            
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, ensure_ascii=False, indent=2)
            logger.info(f"详细结果已保存: {detailed_path}")
            
            # 2. 计算并保存统计信息
            stats = self.calculate_statistics(results)
            stats_path = self._get_safe_file_path(f"{filename}_stats")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"统计信息已保存: {stats_path}")
            
            # 3. 生成并保存分析报告
            analysis = self.generate_analysis(results, stats)
            analysis_path = self._get_safe_file_path(f"{filename}_analysis")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"分析报告已保存: {analysis_path}")
            
            return stats
        
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """计算统计信息（修复相似度统计、新增失败率）
        :param results: 测试结果列表
        :return: 统计字典
"""
        if not results:
            return {}
        
        # 过滤失败用例（仅统计成功用例的性能指标）
        successful_results = [r for r in results if not r.is_failed]
        total_count = len(results)
        success_count = len(successful_results)
        fail_count = total_count
"""计算平均相似度（修复仅取第一个文档的问题）
"""all_similarities = []
        for r in results:
            if r.retrieved_docs:
                # 收集所有返回文档的相似度
                similarities = [doc.get("similarity", 0.0) for doc in r.retrieved_docs]
                all_similarities.extend(similarities)
        
        if not all_similarities:
            return 0.0
        return sum(all_similarities) / len(all_similarities)
    
    def generate_analysis(self, results: List[TestResult], stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成分析报告（增强分析维度、优化建议）
        :param results: 测试结果列表
        :param stats: 统计信息
        :return: 分析报告
"""
        analysis = {
            "overview": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": stats["summary"]["total_test_cases"],
                "success_rate": stats["summary"]["success_rate"],
                "key_performance": {
                    "avg_retrieval_time": stats["performance"].get("avg_retrieval_time", 0.0),
                    "hit_at_1": stats["performance"].get("hit_at_1", 0.0),
                    "avg_mrr": stats["performance"].get("avg_mrr", 0.0)
                }
            },
            "issues": [],
            "recommendations": [],
            "detailed_analysis": {
                "by_query_type": {},
                "by_difficulty": {},
                "by_retrieval_method": {},
                "failed_cases_analysis": {}
            }
        }
        
        # 1. 识别核心问题
        performance = stats.get("performance", {})
        if performance.get("hit_at_1", 0.0) < 0.3:
            analysis["issues"].append("整体Hit@1命中率过低(<30%)，检索准确性严重不足")
        elif performance.get("hit_at_1", 0.0) < 0.5:
            analysis["issues"].append("整体Hit@1命中率偏低(<50%)，需优化检索策略")
        
        if performance.get("avg_retrieval_time", 0.0) > 1.0:
            analysis["issues"].append("平均查询时间过长(>1s)，性能不满足要求")
        elif performance.get("avg_retrieval_time", 0.0) > 0.5:
            analysis["issues"].append("平均查询时间偏长(>0.5s)，可优化性能")
        
        if performance.get("empty_result_rate", 0.0) > 0.2:
            analysis["issues"].append("空结果占比过高(>20%)，需检查数据或查询逻辑")
        
        if performance.get("avg_mrr", 0.0) < 0.3:
            analysis["issues"].append("平均倒数排名过低(<0.3)，相关结果排名靠后")
        
        if stats["summary"]["fail_rate"] > 0.1:
            analysis["issues"].append(f"测试用例失败率过高({stats['summary']['fail_rate']*100:.1f}%)，需修复代码逻辑")
        
        # 2. 按查询类型分析
        query_type_stats = defaultdict(lambda: {
            "count": 0, "hit_at_1": 0.0, "avg_time": 0.0, "avg_mrr": 0.0, "fail_count": 0
        })
        for r in results:
            qt = r.query_type
            query_type_stats[qt]["count"] += 1
            if r.is_failed:
                query_type_stats[qt]["fail_count"] += 1
            else:
                query_type_stats[qt]["hit_at_1"] += 1 if r.hit_at_k.get(1, False) else 0
                query_type_stats[qt]["avg_time"] += r.retrieval_time
                query_type_stats[qt]["avg_mrr"] += r.mrr
        
        # 计算平均值
        for qt, data in query_type_stats.items():
            count = data["count"]
            success_count = count
"""生成针对性优化建议
"""
        performance = stats.get("performance", {})
        summary = stats.get("summary", {})
        query_type_analysis = analysis["detailed_analysis"]["by_query_type"]
        
        # 通用建议
        analysis["recommendations"].extend([
            "定期维护向量数据库，清理低质量/无效文档",
            "使用A/B测试验证不同embedding模型/检索策略的效果",
            "增加单元测试覆盖核心指标计算逻辑",
            "监控生产环境的检索性能和准确性指标",
            "优化测试数据集，补充更多边界场景用例（如空查询、超长查询）"
        ])
        
        # 命中率优化建议
        if performance.get("hit_at_1", 0.0) < 0.5:
            analysis["recommendations"].extend([
                "优化embedding模型：尝试BGE-M3、text-embedding-3-large等高质量模型",
                "调整文档分块策略：避免过大/过小的分块，确保语义完整性",
                "增强查询扩展：使用同义词、相关词丰富查询向量",
                "引入重排序机制：使用交叉编码器对初筛结果重排序",
                "优化元数据过滤逻辑：针对信用代码、地址等特殊查询类型增强精确匹配"
            ])
        
        # 性能优化建议
        if performance.get("avg_retrieval_time", 0.0) > 0.5:
            analysis["recommendations"].extend([
                "优化向量数据库索引：调整HNSW参数(m=16-32, ef_construction=200-400)",
                "启用查询缓存：缓存高频查询结果，设置合理的过期时间",
                "模型优化：使用量化模型、GPU加速或模型蒸馏减小推理耗时",
                "批量处理：对批量查询采用异步/并行处理方式",
                "减少不必要的计算：优化关键词提取和混合检索的权重计算逻辑"
            ])
        
        # 失败率优化建议
        if summary.get("fail_rate", 0.0) > 0.1:
            analysis["recommendations"].extend([
                "修复异常用例：针对失败的错误类型逐一排查（如文档ID格式、网络超时）",
                "增强代码鲁棒性：添加更多参数校验和异常捕获逻辑",
                "优化资源管理：确保数据库连接/模型资源正确释放",
                "增加重试机制：对临时失败的查询（如网络波动）添加自动重试",
                "完善日志输出：为失败用例添加更详细的上下文信息，便于排查"
            ])
        
        # 针对特定查询类型的优化建议
        for qtype, data in query_type_analysis.items():
            if data["hit_at_1"] < 0.4:
                analysis["recommendations"].append(
                    f"优化{qtype}类型查询：当前Hit@1命中率仅{data['hit_at_1']*100:.1f}%，建议针对性调整检索策略（如增强关键词匹配、优化元数据过滤）"
                )
            if data["fail_rate"] > 0.2:
                analysis["recommendations"].append(
                    f"降低{qtype}类型查询失败率：当前失败率{data['fail_rate']*100:.1f}%，建议重点排查该类型查询的参数合法性和资源占用情况"
                )
    
    def print_summary(self, stats: Dict[str, Any], analysis: Dict[str, Any]):
        """打印格式化的总结报告
"""
        print("\n" + "="*80)
        print("📊 向量检索测试评估总结报告")
        print("="*80)
        
        # 基础统计
        summary = stats.get("summary", {})
        print(f"\n🔍 测试概览:")
        print(f"   总测试用例数: {summary.get('total_test_cases', 0)}")
        print(f"   成功执行数: {summary.get('successful_cases', 0)} ({summary.get('success_rate', 0)*100:.1f}%)")
        print(f"   执行失败数: {summary.get('failed_cases', 0)} ({summary.get('fail_rate', 0)*100:.1f}%)")
        
        # 性能指标
        performance = stats.get("performance", {})
        if performance:
            print(f"\n⚡ 性能指标:")
            print(f"   平均查询耗时: {performance.get('avg_retrieval_time', 0):.3f}s")
            print(f"   空结果占比: {performance.get('empty_result_rate', 0)*100:.1f}%")
            print(f"   平均相似度: {performance.get('avg_similarity', 0):.3f}")
            
            print(f"\n🎯 准确性指标:")
            for k in [1,3,5,10,15]:
                hit_rate = performance.get(f"hit_at_{k}", 0)
                precision = performance.get(f"precision_at_{k}", 0)
                recall = performance.get(f"recall_at_{k}", 0)
                print(f"   Hit@{k}: {hit_rate*100:.1f}% | 精确率@{k}: {precision*100:.1f}% | 召回率@{k}: {recall*100:.1f}%")
            print(f"   平均MRR: {performance.get('avg_mrr', 0)*100:.1f}%")
        
        # 按查询类型统计
        query_type_analysis = analysis["detailed_analysis"]["by_query_type"]
        if query_type_analysis:
            print(f"\n📋 按查询类型统计（前5种）:")
            sorted_qtypes = sorted(query_type_analysis.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            for qtype, data in sorted_qtypes:
                print(f"   {qtype}:")
                print(f"     用例数: {data['count']} | 失败率: {data['fail_rate']*100:.1f}%")
                print(f"     Hit@1: {data['hit_at_1']*100:.1f}% | 平均耗时: {data['avg_time']:.3f}s")
        
        # 问题和建议
        if analysis.get("issues"):
            print(f"\n⚠️ 发现的核心问题:")
            for idx, issue in enumerate(analysis["issues"], 1):
                print(f"   {idx}. {issue}")
        
        if analysis.get("recommendations"):
            print(f"\n💡 优化建议 (前10条):")
            for idx, rec in enumerate(analysis["recommendations"][:10], 1):
                print(f"   {idx}. {rec}")
        
        print("\n" + "="*80)

def main():
    """优化后的主函数
"""
    parser = argparse.ArgumentParser(description="优化版向量数据库测试评估工具")
    parser.add_argument("--test_data", type=str, required=True, help="测试集JSON文件路径")
    parser.add_argument("--db_path", type=str, default="/tmp/chroma_db_dsw", help="ChromaDB数据库路径")
    parser.add_argument("--output_dir", type=str, default="optimized_test_results", help="输出目录")
    parser.add_argument("--max_cases", type=int, default=None, help="最大测试用例数")
    parser.add_argument("--top_k", type=int, default=20, help="检索返回的Top-K数量")
    parser.add_argument("--no_progress", action="store_true", help="禁用进度条")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有输出文件")
    
    args = parser.parse_args()
    
    try:
        # 初始化评估器配置
        eval_config = {
            "max_test_cases": args.max_cases,
            "top_k": args.top_k,
            "enable_progress_bar": not args.no_progress,
            "overwrite_output": args.overwrite
        }
        
        # 创建评估器
        evaluator = OptimizedTestEvaluator(
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            config=eval_config
        )
        
        # 初始化查询引擎
        logger.info("初始化向量查询引擎...")
        query_engine = OptimizedVectorDBQuery(db_path=args.db_path)
        
        # 测试连接
        logger.info("测试连接...")
        conn_test = query_engine.test_connection()
        logger.info(f"连接测试结果: {json.dumps(conn_test, ensure_ascii=False, indent=2)}")
        
        # 检查核心连接状态
        if not conn_test.get("model_loaded"):
            raise RuntimeError("查询引擎初始化失败：Embedding模型加载异常")
        if not conn_test.get("db_connected"):
            raise RuntimeError("查询引擎初始化失败：ChromaDB数据库连接异常")
        if not conn_test.get("collection_exists"):
            logger.warning(f"警告：指定的集合{query_engine.collection_name}不存在，可能影响测试结果")
        
        # 运行测试
        results = evaluator.run_test(query_engine)
        
        # 保存结果
        stats = evaluator.save_results(results, "vector_search")
        
        # 生成分析报告
        analysis = evaluator.generate_analysis(results, stats)
        
        # 打印总结
        evaluator.print_summary(stats, analysis)
        
        logger.info("✅ 测试评估流程全部完成!")
        
    except Exception as e:
        logger.error(f"测试评估过程中发生致命错误: {e}")
        logger.error(traceback.format_exc())
        raise SystemExit(1)

if __name__ == "__main__":
    main()