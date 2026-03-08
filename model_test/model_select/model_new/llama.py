import torch
import json
import numpy as np
import faiss
import os
import re
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from typing import List, Dict, Tuple, Optional, Any
import gc

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类
"""
    def __init__(self, llm_name, llm_local_path, embedding_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path
        self.embedding_local_path = embedding_local_path

# ====================== 测试配置 ======================
class TestConfig:
    """测试配置类
"""
    def __init__(self):
        # Llama候选大模型配置（同量级比较）
        self.llm_candidates = [
            {
                "llm_name": "Llama-3.2-3B-Instruct",
                "llm_local_path": "/mnt/workspace/data/modelscope/cache/LLM-Research/Llama-3.2-3B-Instruct",
                "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-
"""设置输出目录（包含模型名称）
"""timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = self.llm_config['llm_name'].replace("/", "_").replace("\\", "_")
        output_dir = os.path.join(
            self.config.output_dir_base, 
            f"{model_name_clean}_{timestamp}"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        return output_dir
    
    def run_llm_selection_tests(self):
        """运行Llama模型选型测试
"""
        print(f"\n{'='*60}")
        print(f"开始测试Llama模型: {self.llm_config['llm_name']}")
        print(f"模型路径: {self.llm_config['llm_local_path']}")
        print(f"{'='*60}")
        
        # 加载模型
        tokenizer, llm_model, embedding_models = ModelManager.load_local_models(self.llm_config)
        if llm_model is None:
            print(f"❌ Llama模型 {self.llm_config['llm_name']} 加载失败，跳过测试")
            return None
        
        # 构建向量索引
        print("\n正在构建FAISS向量索引...")
        index, enhanced_docs = build_vector_index(embedding_models, self.knowledge_docs, 
                                                  batch_size=self.config.batch_size)
        if index is None:
            print("⚠️  向量索引构建失败，场景2将无法测试")
        
        # 运行测试
        scenario1_results = []
        scenario2_results = []
        
        for idx, test_case in enumerate(self.test_cases):
            question = test_case["question"]
            reference_answer = test_case["reference_answer"]
            relevant_docs = test_case["relevant_docs"]
            
            print(f"\n=== 测试用例 {idx+1}/{len(self.test_cases)} ===")
            print(f"问题：{question}")
            
            # 场景1：无提示词直接回答
            print("\n--
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
    
    def _generate_summary_text(self, summary):
        """生成摘要文本
"""
        text = f"""Llama模型测试结果摘要
模型名称: {self.llm_config['llm_name']}
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
测试用例数量: {summary['test_cases_count']}

场景1测试结果（无提示词）:
"""
        return text

# ====================== 主函数 ======================
def main():
    """主函数：运行Llama模型选型测试
"""
    
    print(f"{'='*60}")
    print("Llama模型选型测试框架")
    print("阶段1：Llama模型测试（学习能力 + 可训练能力）")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有提示词RAG推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建测试配置
    test_config = TestConfig()
    
    if not test_config.llm_candidates:
        print("❌ 未配置候选Llama模型，请检查TestConfig类中的llm_candidates列表")
        return
    
    print(f"共有 {len(test_config.llm_candidates)} 个候选Llama模型")
    
    all_results = []
    
    # 逐个测试每个候选模型
    for llm_config in test_config.llm_candidates:
        print(f"\n{'='*60}")
        print(f"开始测试模型: {llm_config['llm_name']}")
        print(f"{'='*60}")
        
        # 创建测试运行器
        test_runner = ModelTestRunner(test_config, llm_config)
        
        # 运行Llama模型测试
        llm_results = test_runner.run_llm_selection_tests()
        
        if llm_results is not None:
            all_results.append({
                "model_name": llm_config['llm_name'],
                "output_dir": test_runner.output_dir
            })
        
        print(f"\n{'='*60}")
        print(f"✅ 模型 {llm_config['llm_name']} 测试完成")
        print(f"{'='*60}")
    
    # 输出所有测试结果汇总
    print(f"\n{'='*60}")
    print("✅ 所有Llama模型测试完成")
    print(f"{'='*60}")
    
    if all_results:
        print("\n📋 各模型测试结果输出目录:")
        for result in all_results:
            print(f"  模型: {result['model_name']}")
            print(f"  输出目录: {result['output_dir']}")
            print()
    
    print("下一步：根据测试结果分析各Llama模型表现")
    print("然后进行阶段2：Embedding模型选型")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()