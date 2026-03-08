import torch
import json
import numpy as np
import faiss
import os
import re
import sys
import gc
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from difflib import SequenceMatcher

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
"""模型管理器（针对InternLM2优化）"""
    
    @staticmethod
    def get_bnb_config():
        """获取量化参数配置
"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    @staticmethod
    def load_local_models(config):
        """加载本地模型（针对InternLM2优化）
        返回: (tokenizer, llm_model, embedding_models)
"""
        print(f"正在加载本地大模型：{config['llm_name']}")
        
        try:
            # 1. 加载大模型tokenizer（针对InternLM2优化）
            tokenizer = AutoTokenizer.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                padding_side="left",  # InternLM2使用左侧填充
                use_fast=False,       # 使用慢速但更准确的tokenizer
                local_files_only=True
            )
            
            # 设置pad_token（InternLM2特定）
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.unk_token
            
            print(f"✅ InternLM2 Tokenizer加载成功，pad_token: {tokenizer.pad_token}")
            
            # 2. 加载大模型
            llm_model = AutoModelForCausalLM.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                device_map="auto",
                quantization_config=ModelManager.get_bnb_config(),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            llm_model.eval()
            print(f"✅ InternLM2模型加载成功")
        except Exception as e:
            print(f"❌ 大模型加载失败，错误信息：{e}")
            return None, None, None
        
        # 3. 加载Embedding模型（固定使用BGE-large-zh-
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
    
    def _generate_brief_statistics(self, llm_results):
        """生成简要统计报告
"""
        stats_file = os.path.join(self.output_dir, f"{self.model_name}_brief_statistics.txt")
        
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"{self.model_name} 模型测试简要统计\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"模型名称: {llm_results['model_name']}\n")
            f.write(f"测试时间: {llm_results['test_time']}\n")
            f.write(f"测试用例数: {llm_results['test_cases_count']}\n\n")
            
            f.write("📊 测试结果摘要:\n")
            f.write(f"   场景1平均准确率: {llm_results['summary']['scenario1_avg_accuracy']:.4f}\n")
            f.write(f"   场景1平均质量分: {llm_results['summary']['scenario1_avg_quality']:.4f}\n")
            f.write(f"   场景1平均回答长度: {llm_results['summary']['scenario1_avg_answer_length']:.1f}\n\n")
            
            f.write(f"   场景2平均准确率: {llm_results['summary']['scenario2_avg_accuracy']:.4f}\n")
            f.write(f"   场景2平均召回率: {llm_results['summary']['scenario2_avg_recall']:.4f}\n")
            f.write(f"   场景2平均质量分: {llm_results['summary']['scenario2_avg_quality']:.4f}\n")
            f.write(f"   场景2平均回答长度: {llm_results['summary']['scenario2_avg_answer_length']:.1f}\n")
            f.write(f"   场景2平均检索文档数: {llm_results['summary']['scenario2_avg_retrieved_count']:.1f}\n\n")
            
            # 成功/失败统计
            scenario1_success = sum(1 for r in llm_results['scenario1_results'] if r.get("accuracy", 0) == 1)
            scenario1_fail = len(llm_results['scenario1_results'])
"""主函数：运行模型测试
"""
    
    print(f"{'='*60}")
    print("模型选型第一阶段测试框架")
    print("目标：在固定Embedding模型和向量库下测试大模型表现")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有提示词RAG推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建测试配置
    test_config = TestConfig()
    
    print(f"\n候选模型数量: {len(test_config.llm_candidates)}")
    
    for model_config in test_config.llm_candidates:
        model_name = model_config["llm_name"]
        print(f"\n{'='*60}")
        print(f"开始测试模型: {model_name}")
        print(f"{'='*60}")
        
        # 为每个模型创建独立的测试运行器
        test_runner = ModelTestRunner(test_config, model_name, model_config)
        
        # 运行该模型的测试
        test_runner.run_llm_selection_tests()
        
        print(f"\n✅ {model_name} 模型测试完成")
        
        # 等待一段时间，确保资源释放
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print("✅ 模型测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()