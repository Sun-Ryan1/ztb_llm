#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试评估框架
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from tqdm import tqdm

from vector_db_query_optimized import OptimizedVectorDBQuery, QueryResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s
"""测试结果
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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
"""
        return {
            "test_id": self.test_id,
            "query": self.query,
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "retrieval_method": self.retrieval_method,
            "retrieval_time": self.retrieval_time,
            "expected_docs": len(self.expected_doc_ids),
            "retrieved_docs": len(self.actual_doc_ids),
            "hit_at_1": self.hit_at_k.get(1, False),
            "hit_at_3": self.hit_at_k.get(3, False),
            "hit_at_5": self.hit_at_k.get(5, False),
            "hit_at_10": self.hit_at_k.get(10, False),
            "hit_at_15": self.hit_at_k.get(15, False),
            "precision_at_1": self.precision_at_k.get(1, 0),
            "precision_at_3": self.precision_at_k.get(3, 0),
            "precision_at_5": self.precision_at_k.get(5, 0),
            "precision_at_10": self.precision_at_k.get(10, 0),
            "precision_at_15": self.precision_at_k.get(15, 0),
            "recall_at_1": self.recall_at_k.get(1, 0),
            "recall_at_3": self.recall_at_k.get(3, 0),
            "recall_at_5": self.recall_at_k.get(5, 0),
            "recall_at_10": self.recall_at_k.get(10, 0),
            "recall_at_15": self.recall_at_k.get(15, 0),
            "mrr": self.mrr
        }

class FixedTestEvaluator:
    """测试评估器
"""
    
    def __init__(self, test_data_path: str, output_dir: str = "fixed_test_results"):
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载测试数据
        self.test_data = self.load_test_data()
        logger.info(f"加载测试数据: {len(self.test_data['test_cases'])} 个测试用例")
        
        # 配置
        self.config = {
            "max_test_cases": None,  # 默认不限制测试用例数量
            "k_values": [1, 3, 5, 10, 15],
            "enable_progress_bar": True,
            "top_k": 20  # 检索文档数量，增加到20以获取更多结果
        }
    
    def load_test_data(self) -> Dict[str, Any]:
        """加载测试数据
"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _normalize_doc_id(self, doc_id: str) -> str:
        """标准化文档ID，确保格式一致
"""
        if not doc_id:
            return ""
        # 移除可能的前缀或后缀
        doc_id = doc_id.strip()
        # 转换为统一格式（去除空格，转为小写，处理可能的编码问题）
        return doc_id.replace(" ", "").lower()
    
    def calculate_metrics(self, expected_ids: List[str], actual_ids: List[str]) -> tuple:
        """计算指标
"""
        hit_at_k = {k: False for k in self.config["k_values"]}
        precision_at_k = {k: 0.0 for k in self.config["k_values"]}
        recall_at_k = {k: 0.0 for k in self.config["k_values"]}
        
        # 标准化预期ID
        normalized_expected = [self._normalize_doc_id(id) for id in expected_ids if id]
        # 标准化实际ID
        normalized_actual = [self._normalize_doc_id(id) for id in actual_ids if id]
        
        for k in self.config["k_values"]:
            top_k = normalized_actual[:k]
            
            if not normalized_expected:
                hit_at_k[k] = len(top_k) == 0
                precision_at_k[k] = 1.0 if len(top_k) == 0 else 0.0
                recall_at_k[k] = 1.0 if len(top_k) == 0 else 0.0
            else:
                hits = sum(1 for doc_id in top_k if doc_id in normalized_expected)
                hit_at_k[k] = hits > 0
                precision_at_k[k] = hits / k if k > 0 else 0.0
                recall_at_k[k] = hits / len(normalized_expected) if normalized_expected else 0.0
        
        # 计算MRR
        mrr = 0.0
        for i, doc_id in enumerate(normalized_actual):
            if doc_id in normalized_expected:
                mrr = 1.0 / (i + 1)
                break
        
        return hit_at_k, precision_at_k, recall_at_k, mrr
    
    def run_test(self, query_engine: OptimizedVectorDBQuery, 
                test_cases: List[Dict] = None) -> List[TestResult]:
        """运行测试
"""
        if test_cases is None:
            test_cases = self.test_data["test_cases"]
        
        # 限制测试用例数量（如果配置了限制）
        if self.config["max_test_cases"] is not None:
            test_cases = test_cases[:self.config["max_test_cases"]]
        
        logger.info(f"开始测试，测试用例数: {len(test_cases)}")
        
        results = []
        
        # 使用进度条
        pbar = tqdm(test_cases, desc="测试进度") if self.config["enable_progress_bar"] else test_cases
        
        for test_case in pbar:
            try:
                test_id = test_case.get("test_id", "unknown")
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
"""保存结果并计算统计
"""
        # 保存详细结果
        detailed_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            # 简化输出，只保存必要信息
            simplified_results = []
            for r in results:
                simplified = r.to_dict()
                # 添加前3个检索到的文档预览
                if r.retrieved_docs:
                    simplified["retrieved_docs_preview"] = [
                        {
                            "id": doc.get("id", ""),
                            "content_preview": doc.get("content_preview", ""),
                            "similarity": doc.get("similarity", 0)
                        }
                        for doc in r.retrieved_docs[:3]
                    ]
                simplified_results.append(simplified)
            
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        # 计算统计
        stats = self.calculate_statistics(results)
        
        # 保存统计
        stats_path = os.path.join(self.output_dir, f"{filename}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 生成分析报告
        analysis = self.generate_analysis(results, stats)
        analysis_path = os.path.join(self.output_dir, f"{filename}_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {detailed_path}")
        logger.info(f"统计已保存到: {stats_path}")
        logger.info(f"分析已保存到: {analysis_path}")
        
        return stats
    
    def calculate_statistics(self, results: List[TestResult]) -> Dict[str, Any]:
        """计算统计信息
"""
        if not results:
            return {}
        
        stats = {
            "total_queries": len(results),
            "total_time": sum(r.retrieval_time for r in results),
            "avg_time": sum(r.retrieval_time for r in results) / len(results),
            "hit_at_1": sum(1 for r in results if r.hit_at_k.get(1, False)) / len(results),
            "hit_at_3": sum(1 for r in results if r.hit_at_k.get(3, False)) / len(results),
            "hit_at_5": sum(1 for r in results if r.hit_at_k.get(5, False)) / len(results),
            "hit_at_10": sum(1 for r in results if r.hit_at_k.get(10, False)) / len(results),
            "hit_at_15": sum(1 for r in results if r.hit_at_k.get(15, False)) / len(results),
            "precision_at_1": sum(r.precision_at_k.get(1, 0) for r in results) / len(results),
            "precision_at_3": sum(r.precision_at_k.get(3, 0) for r in results) / len(results),
            "precision_at_5": sum(r.precision_at_k.get(5, 0) for r in results) / len(results),
            "precision_at_10": sum(r.precision_at_k.get(10, 0) for r in results) / len(results),
            "precision_at_15": sum(r.precision_at_k.get(15, 0) for r in results) / len(results),
            "recall_at_1": sum(r.recall_at_k.get(1, 0) for r in results) / len(results),
            "recall_at_3": sum(r.recall_at_k.get(3, 0) for r in results) / len(results),
            "recall_at_5": sum(r.recall_at_k.get(5, 0) for r in results) / len(results),
            "recall_at_10": sum(r.recall_at_k.get(10, 0) for r in results) / len(results),
            "recall_at_15": sum(r.recall_at_k.get(15, 0) for r in results) / len(results),
            "mrr": sum(r.mrr for r in results) / len(results),
            "successful_queries": len([r for r in results if r.actual_doc_ids]),
            "empty_results": len([r for r in results if not r.actual_doc_ids]),
            "avg_similarity": sum(r.retrieved_docs[0]['similarity'] if r.retrieved_docs else 0 for r in results) / len(results)
        }
        
        return stats
    
    def generate_analysis(self, results: List[TestResult], stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成详细分析报告
"""
        analysis = {
            "performance_summary": {
                "avg_query_time": stats.get("avg_time", 0),
                "hit_rate_at_1": stats.get("hit_at_1", 0),
                "hit_rate_at_5": stats.get("hit_at_5", 0),
                "hit_rate_at_10": stats.get("hit_at_10", 0),
                "precision_at_1": stats.get("precision_at_1", 0),
                "precision_at_5": stats.get("precision_at_5", 0),
                "recall_at_5": stats.get("recall_at_5", 0),
                "mrr": stats.get("mrr", 0),
                "success_rate": stats.get("successful_queries", 0) / stats.get("total_queries", 1) if stats.get("total_queries", 0) > 0 else 0
            },
            "query_analysis": {
                "by_type": defaultdict(lambda: {
                    "count": 0,
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "hit_at_10": 0,
                    "precision_at_1": 0,
                    "precision_at_5": 0,
                    "recall_at_5": 0,
                    "mrr": 0,
                    "avg_time": 0,
                    "avg_similarity": 0,
                    "retrieval_methods": defaultdict(int)
                }),
                "by_difficulty": defaultdict(lambda: {
                    "count": 0,
                    "hit_at_1": 0,
                    "hit_at_5": 0,
                    "avg_time": 0
                })
            },
            "retrieval_method_analysis": defaultdict(lambda: {
                "count": 0,
                "hit_at_1": 0,
                "avg_time": 0,
                "avg_similarity": 0
            }),
            "issues": [],
            "recommendations": [],
            "detailed_issues": {
                "low_hit_rate_types": [],
                "slow_query_types": [],
                "empty_result_queries": []
            }
        }
        
        # 按查询类型和难度分析
        for result in results:
            qtype = result.query_type
            diff = result.difficulty
            method = result.retrieval_method
            
            # 按查询类型统计
            analysis["query_analysis"]["by_type"][qtype]["count"] += 1
            analysis["query_analysis"]["by_type"][qtype]["hit_at_1"] += 1 if result.hit_at_k.get(1, False) else 0
            analysis["query_analysis"]["by_type"][qtype]["hit_at_5"] += 1 if result.hit_at_k.get(5, False) else 0
            analysis["query_analysis"]["by_type"][qtype]["hit_at_10"] += 1 if result.hit_at_k.get(10, False) else 0
            analysis["query_analysis"]["by_type"][qtype]["precision_at_1"] += result.precision_at_k.get(1, 0)
            analysis["query_analysis"]["by_type"][qtype]["precision_at_5"] += result.precision_at_k.get(5, 0)
            analysis["query_analysis"]["by_type"][qtype]["recall_at_5"] += result.recall_at_k.get(5, 0)
            analysis["query_analysis"]["by_type"][qtype]["mrr"] += result.mrr
            analysis["query_analysis"]["by_type"][qtype]["avg_time"] += result.retrieval_time
            analysis["query_analysis"]["by_type"][qtype]["retrieval_methods"][method] += 1
            
            # 统计平均相似度
            if result.retrieved_docs:
                analysis["query_analysis"]["by_type"][qtype]["avg_similarity"] += result.retrieved_docs[0]['similarity']
            
            # 按难度统计
            analysis["query_analysis"]["by_difficulty"][diff]["count"] += 1
            analysis["query_analysis"]["by_difficulty"][diff]["hit_at_1"] += 1 if result.hit_at_k.get(1, False) else 0
            analysis["query_analysis"]["by_difficulty"][diff]["hit_at_5"] += 1 if result.hit_at_k.get(5, False) else 0
            analysis["query_analysis"]["by_difficulty"][diff]["avg_time"] += result.retrieval_time
            
            # 按检索方法统计
            analysis["retrieval_method_analysis"][method]["count"] += 1
            analysis["retrieval_method_analysis"][method]["hit_at_1"] += 1 if result.hit_at_k.get(1, False) else 0
            analysis["retrieval_method_analysis"][method]["avg_time"] += result.retrieval_time
            if result.retrieved_docs:
                analysis["retrieval_method_analysis"][method]["avg_similarity"] += result.retrieved_docs[0]['similarity']
        
        # 计算平均值
        for qtype in analysis["query_analysis"]["by_type"]:
            count = analysis["query_analysis"]["by_type"][qtype]["count"]
            if count > 0:
                analysis["query_analysis"]["by_type"][qtype]["hit_at_1"] /= count
                analysis["query_analysis"]["by_type"][qtype]["hit_at_5"] /= count
                analysis["query_analysis"]["by_type"][qtype]["hit_at_10"] /= count
                analysis["query_analysis"]["by_type"][qtype]["precision_at_1"] /= count
                analysis["query_analysis"]["by_type"][qtype]["precision_at_5"] /= count
                analysis["query_analysis"]["by_type"][qtype]["recall_at_5"] /= count
                analysis["query_analysis"]["by_type"][qtype]["mrr"] /= count
                analysis["query_analysis"]["by_type"][qtype]["avg_time"] /= count
                analysis["query_analysis"]["by_type"][qtype]["avg_similarity"] /= count
        
        for diff in analysis["query_analysis"]["by_difficulty"]:
            count = analysis["query_analysis"]["by_difficulty"][diff]["count"]
            if count > 0:
                analysis["query_analysis"]["by_difficulty"][diff]["hit_at_1"] /= count
                analysis["query_analysis"]["by_difficulty"][diff]["hit_at_5"] /= count
                analysis["query_analysis"]["by_difficulty"][diff]["avg_time"] /= count
        
        for method in analysis["retrieval_method_analysis"]:
            count = analysis["retrieval_method_analysis"][method]["count"]
            if count > 0:
                analysis["retrieval_method_analysis"][method]["hit_at_1"] /= count
                analysis["retrieval_method_analysis"][method]["avg_time"] /= count
                analysis["retrieval_method_analysis"][method]["avg_similarity"] /= count
        
        # 识别问题
        if stats.get("hit_at_1", 0) < 0.3:
            analysis["issues"].append("整体命中率过低，可能需要检查向量化模型或文档质量")
        elif stats.get("hit_at_1", 0) < 0.5:
            analysis["issues"].append("整体命中率一般，有较大优化空间")
        
        if stats.get("avg_time", 0) > 1.0:
            analysis["issues"].append("查询时间过长，可能需要优化索引或批量处理")
        elif stats.get("avg_time", 0) > 0.5:
            analysis["issues"].append("查询时间偏长，可考虑优化检索流程")
        
        if stats.get("empty_results", 0) > len(results) * 0.5:
            analysis["issues"].append("过多查询返回空结果，可能需要重新构建向量数据库")
        elif stats.get("empty_results", 0) > len(results) * 0.2:
            analysis["issues"].append("较多查询返回空结果，可优化查询扩展或向量模型")
        
        if stats.get("mrr", 0) < 0.3:
            analysis["issues"].append("平均倒数排名偏低，说明相关结果排名靠后，需要优化重排序策略")
        
        # 识别具体问题的查询类型
        for qtype, data in analysis["query_analysis"]["by_type"].items():
            if data["hit_at_1"] < 0.2:
                analysis["detailed_issues"]["low_hit_rate_types"].append({
                    "type": qtype,
                    "hit_at_1": data["hit_at_1"],
                    "count": data["count"],
                    "precision_at_1": data["precision_at_1"],
                    "recall_at_5": data["recall_at_5"]
                })
            elif data["hit_at_1"] < 0.4:
                analysis["detailed_issues"]["low_hit_rate_types"].append({
                    "type": qtype,
                    "hit_at_1": data["hit_at_1"],
                    "count": data["count"],
                    "precision_at_1": data["precision_at_1"],
                    "recall_at_5": data["recall_at_5"]
                })
            
            if data["avg_time"] > 1.5:
                analysis["detailed_issues"]["slow_query_types"].append({
                    "type": qtype,
                    "avg_time": data["avg_time"],
                    "count": data["count"],
                    "hit_at_1": data["hit_at_1"]
                })
            elif data["avg_time"] > 0.8:
                analysis["detailed_issues"]["slow_query_types"].append({
                    "type": qtype,
                    "avg_time": data["avg_time"],
                    "count": data["count"],
                    "hit_at_1": data["hit_at_1"]
                })
        
        # 识别空结果的查询
        empty_result_queries = []
        for result in results:
            if not result.actual_doc_ids:
                empty_result_queries.append({
                    "test_id": result.test_id,
                    "query": result.query,
                    "query_type": result.query_type,
                    "difficulty": result.difficulty
                })
        analysis["detailed_issues"]["empty_result_queries"] = empty_result_queries
        
        # 生成详细建议
        # 1. 整体性能建议
        if stats.get("hit_at_1", 0) < 0.5:
            analysis["recommendations"].append("考虑使用更先进的embedding模型，如bge-m3或text-embedding-3-large，提高向量表示质量")
            analysis["recommendations"].append("尝试增加检索候选数量（top_k）到20-30，为后续重排序提供更多优质候选")
            analysis["recommendations"].append("检查文档分块策略是否合适，过大或过小的块都会影响检索效果")
            analysis["recommendations"].append("考虑使用指令式embedding，如在查询前添加'为这个查询检索相关文档：'，增强向量表示的针对性")
        
        if stats.get("avg_time", 0) > 0.5:
            analysis["recommendations"].append("确保查询缓存已启用，调整缓存大小到2000-5000条，过期时间设置为1-2小时")
            analysis["recommendations"].append("考虑使用轻量级模型或优化模型加载方式，如使用model_kwargs={'device_map':'auto'}")
            analysis["recommendations"].append("优化ChromaDB索引参数，尝试调整hnsw:m为16-32，hnsw:ef_construction为200-400")
            analysis["recommendations"].append("考虑使用批量查询或异步查询方式，减少并发请求延迟")
        
        # 2. 针对不同查询类型的具体建议
        for qtype, data in analysis["query_analysis"]["by_type"].items():
            if data["hit_at_1"] < 0.4:
                if qtype == "credit_code":
                    analysis["recommendations"].append("信用代码查询：")
                    analysis["recommendations"].append("
"""打印总结报告
"""
        print("\n" + "="*80)
        print("测试总结报告")
        print("="*80)
        
        print(f"\n📊 性能指标:")
        print(f"   总查询数: {stats.get('total_queries', 0)}")
        print(f"   平均查询时间: {stats.get('avg_time', 0):.3f}s")
        print(f"   平均相似度: {stats.get('avg_similarity', 0):.3f}")
        print(f"   命中率@1: {stats.get('hit_at_1', 0)*100:.1f}%")
        print(f"   命中率@3: {stats.get('hit_at_3', 0)*100:.1f}%")
        print(f"   命中率@5: {stats.get('hit_at_5', 0)*100:.1f}%")
        print(f"   命中率@10: {stats.get('hit_at_10', 0)*100:.1f}%")
        print(f"   命中率@15: {stats.get('hit_at_15', 0)*100:.1f}%")
        print(f"   精确率@1: {stats.get('precision_at_1', 0)*100:.1f}%")
        print(f"   精确率@5: {stats.get('precision_at_5', 0)*100:.1f}%")
        print(f"   精确率@10: {stats.get('precision_at_10', 0)*100:.1f}%")
        print(f"   召回率@5: {stats.get('recall_at_5', 0)*100:.1f}%")
        print(f"   召回率@10: {stats.get('recall_at_10', 0)*100:.1f}%")
        print(f"   MRR: {stats.get('mrr', 0)*100:.1f}%")
        print(f"   成功查询数: {stats.get('successful_queries', 0)}")
        print(f"   空结果数: {stats.get('empty_results', 0)}")
        
        if analysis.get("issues"):
            print(f"\n⚠️ 发现的问题:")
            for issue in analysis["issues"]:
                print(f"   • {issue}")
        
        if analysis.get("recommendations"):
            print(f"\n💡 优化建议:")
            for rec in analysis["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "="*80)

def main():
    """主函数
"""
    parser = argparse.ArgumentParser(description=" 向量数据库测试评估")
    parser.add_argument("--test_data", type=str, required=True,
                       help="测试集JSON文件路径")
    parser.add_argument("--db_path", type=str, default="/tmp/chroma_db_dsw",
                       help="ChromaDB数据库路径")
    parser.add_argument("--output_dir", type=str, default="fixed_test_results",
                       help="输出目录")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="最大测试用例数，默认无限制")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("开始 向量数据库测试评估")
    logger.info("="*80)
    
    # 初始化评估器
    evaluator = FixedTestEvaluator(args.test_data, args.output_dir)
    evaluator.config["max_test_cases"] = args.max_cases
    
    # 初始化查询器
    logger.info("初始化查询器...")
    try:
        query_engine = OptimizedVectorDBQuery(db_path=args.db_path)
        
        # 测试连接
        logger.info("测试连接...")
        test_result = query_engine.test_connection()
        logger.info(f"连接测试结果: {json.dumps(test_result, ensure_ascii=False)}")
        
        if not test_result["model_loaded"]:
            logger.error("模型加载失败!")
            return
        
        if not test_result["db_connected"]:
            logger.error("数据库连接失败!")
            return
        
        # 获取集合统计
        stats = query_engine.get_collection_stats()
        logger.info(f"集合统计: {json.dumps(stats, ensure_ascii=False)}")
        
        # 运行测试
        results = evaluator.run_test(query_engine)
        
        # 保存结果
        stats = evaluator.save_results(results, "vector_search")
        
        # 生成分析
        analysis = evaluator.generate_analysis(results, stats)
        
        # 打印总结
        evaluator.print_summary(stats, analysis)
        
        # 关闭查询器
        query_engine.close()
        
        logger.info("✅ 测试完成!")
        logger.info(f"结果保存在: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()