import torch
import json
import numpy as np
import faiss
import os
import re
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import sys

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类
"""
    def __init__(self, llm_name, llm_local_path, embedding_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path
        self.embedding_local_path = embedding_local_path

# ====================== Qwen模型配置 ======================
class QwenTestConfig:
    """Qwen模型测试配置
"""
    def __init__(self):
        # Qwen模型配置
        self.llm_config = {
            "llm_name": "Qwen2.5-7B-Instruct",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen2-5-7B-Instruct",
            "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-
"""计算测试结果摘要
"""
        # 场景1统计
        scenario1_accuracies = [r.get("accuracy", 0) for r in scenario1_results if "accuracy" in r]
        scenario1_quality_scores = [r.get("quality_score", 0) for r in scenario1_results if "quality_score" in r]
        scenario1_answer_lengths = [r.get("answer_length", 0) for r in scenario1_results if "answer_length" in r]
        
        # 场景2统计
        scenario2_accuracies = [r.get("accuracy", 0) for r in scenario2_results if "accuracy" in r]
        scenario2_recalls = [r.get("recall_score", 0) for r in scenario2_results if "recall_score" in r]
        scenario2_quality_scores = [r.get("quality_score", 0) for r in scenario2_results if "quality_score" in r]
        scenario2_answer_lengths = [r.get("answer_length", 0) for r in scenario2_results if "answer_length" in r]
        scenario2_retrieved_counts = [r.get("retrieved_count", 0) for r in scenario2_results if "retrieved_count" in r]
        
        return {
            "test_cases_count": len(scenario1_results),
            "scenario1_avg_accuracy": np.mean(scenario1_accuracies) if scenario1_accuracies else 0,
            "scenario1_avg_quality": np.mean(scenario1_quality_scores) if scenario1_quality_scores else 0,
            "scenario1_avg_answer_length": np.mean(scenario1_answer_lengths) if scenario1_answer_lengths else 0,
            "scenario2_avg_accuracy": np.mean(scenario2_accuracies) if scenario2_accuracies else 0,
            "scenario2_avg_recall": np.mean(scenario2_recalls) if scenario2_recalls else 0,
            "scenario2_avg_quality": np.mean(scenario2_quality_scores) if scenario2_quality_scores else 0,
            "scenario2_avg_answer_length": np.mean(scenario2_answer_lengths) if scenario2_answer_lengths else 0,
            "scenario2_avg_retrieved_count": np.mean(scenario2_retrieved_counts) if scenario2_retrieved_counts else 0
        }
    
    def _generate_log_file(self, qwen_results):
        """生成日志文件
"""
        log_content = f"""Qwen模型测试日志
测试时间: {qwen_results['test_time']}
模型名称: {qwen_results['llm_config']['llm_name']}
模型路径: {qwen_results['llm_config']['llm_local_path']}

测试配置:
  测试用例数: {qwen_results['test_config']['test_cases_count']}
  知识库文档数: {qwen_results['test_config']['knowledge_docs_count']}
  最大测试用例数: {qwen_results['test_config']['max_test_cases']}
  批处理大小: {qwen_results['test_config']['batch_size']}
  检索top_k: {qwen_results['test_config']['top_k_retrieval']}
  相似度阈值: {qwen_results['test_config']['similarity_threshold']}

测试结果摘要:
  场景1平均准确率: {qwen_results['summary']['scenario1_avg_accuracy']:.4f}
  场景1平均质量分: {qwen_results['summary']['scenario1_avg_quality']:.4f}
  场景1平均回答长度: {qwen_results['summary']['scenario1_avg_answer_length']:.2f}
  场景2平均准确率: {qwen_results['summary']['scenario2_avg_accuracy']:.4f}
  场景2平均召回率: {qwen_results['summary']['scenario2_avg_recall']:.4f}
  场景2平均质量分: {qwen_results['summary']['scenario2_avg_quality']:.4f}
  场景2平均回答长度: {qwen_results['summary']['scenario2_avg_answer_length']:.2f}
  场景2平均检索文档数: {qwen_results['summary']['scenario2_avg_retrieved_count']:.2f}

详细结果请查看llm_results目录下的JSON文件。
"""
        
        log_file = os.path.join(self.output_dir, "logs", "test_summary.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        
        print(f"✅ 日志文件已生成: {log_file}")

# ====================== 主函数 ======================
def main():
    """主函数：运行Qwen模型测试
"""
    
    print(f"{'='*60}")
    print("Qwen模型测试")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有提示词RAG推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建Qwen测试配置
    test_config = QwenTestConfig()
    
    # 创建测试运行器
    test_runner = QwenModelTestRunner(test_config)
    
    # 运行Qwen模型测试
    test_runner.run_qwen_tests()
    
    print(f"\n{'='*60}")
    print("✅ Qwen模型测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()