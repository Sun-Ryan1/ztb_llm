import torch
import json
import numpy as np
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import gc

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类
"""
    def __init__(self, llm_name, llm_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path

# ====================== Llama模型配置 ======================
class LlamaTestConfig:
    """Llama模型测试配置
"""
    def __init__(self):
        # Llama模型配置
        self.llm_config = {
            "llm_name": "Llama-3.2-3B-Instruct",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/LLM-Research/Llama-3.2-3B-Instruct"
        }
        
        # 测试数据路径
        self.test_data_path = "qa_data/100_qa.json"
        
        # 输出目录
        self.output_dir = f"./llama_model_results_noRAG"
        
        # 测试参数
        self.max_test_cases = 50
        self.batch_size = 16
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.75

# ====================== 模型管理器 ======================
class ModelManager:
    """模型管理器
"""
    
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
        """加载本地模型
        返回: (tokenizer, llm_model)
"""
        print(f"正在加载Llama模型：{config['llm_name']}")
        
        try:
            # 1. 加载Llama模型tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                padding_side="right",
                local_files_only=True
            )
            
            # Llama特殊处理：设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ Llama Tokenizer加载成功")
            
            # 2. 加载Llama模型
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
            print(f"✅ Llama模型 {config['llm_name']} 加载成功")
            return tokenizer, llm_model
        except Exception as e:
            print(f"❌ Llama模型加载失败，错误信息：{e}")
            return None, None
    
    @staticmethod
    def cleanup_models(llm_model):
        """清理模型资源
"""
        if llm_model is not None:
            del llm_model
        
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ 模型资源已清理")

# ====================== 推理函数 ======================
def direct_inference_no_prompt(tokenizer, llm_model, question):
    """场景1：无任何提示词，直接让模型回答问题
"""
    # Llama可能需要不同的格式
    prompt = f"{question}"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 移除问题部分
    if model_answer.startswith(question):
        model_answer = model_answer[len(question):].strip()
    
    return model_answer

def inference_with_professional_prompt(tokenizer, llm_model, question):
    """场景2：使用专业提示词模板
"""
    # 使用专业提示词模板
    prompt = f"""# 角色定位
你是聚焦招投标采购全流程的专业智能问答系统，需严格依据《招标投标法》《政府采购法》等法规，精准解答政策合规、业务操作、物资产品、电子系统操作等领域问题。

# 回答要求
1. 准确性：严格依据相关法规和政策，确保信息准确无误
2. 完整性：全面覆盖问题要点，提供详细的分析和解释
3. 专业性：正确使用专业术语，体现专业知识和分析能力
4. 清晰性：语言流畅，逻辑清晰，结构合理

# 现在请根据你的知识回答问题

问：{question}
答："""inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 从响应中提取模型回答部分
    answer_marker = "答："
    if answer_marker in full_response:
        last_answer_pos = full_response.rfind(answer_marker)
        model_answer = full_response[last_answer_pos + len(answer_marker):].strip()
    else:
        model_answer = full_response.strip()
    
    return model_answer

# ====================== 数据加载器 ======================
def load_qa_data(qa_file_path="qa_data/100_qa.json"):
"""加载QA数据"""try:
        # 加载问答对
        with open(qa_file_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        test_cases = []
        for idx, item in enumerate(qa_data):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            
            if not question or not answer:
                continue
            
            test_cases.append({
                "question": question,
                "reference_answer": answer,
                "scene": item.get("scene", "unknown"),
                "source_table": item.get("source_table", "unknown")
            })
        
        print(f"✅ 加载 {len(test_cases)} 条有效测试用例")
        
        return test_cases
    
    except FileNotFoundError:
        print(f"❌ 未找到文件：{qa_file_path}")
        return []
    except Exception as e:
        print(f"❌ 加载问答对失败：{e}")
        return []

# ====================== 评估函数 ======================
def calculate_accuracy(model_answer, reference_answer, threshold=0.6):
"""计算准确率"""if not model_answer or not reference_answer:
        return 0
    
    if reference_answer in model_answer or model_answer in reference_answer:
        return 1
    
    important_keywords = ["法定代表人", "公司", "地址", "金额", "供应商", "采购方", "中标", "价格", "项目"]
    match_count = 0
    total_keywords = 0
    
    for keyword in important_keywords:
        if keyword in reference_answer:
            total_keywords += 1
            if keyword in model_answer:
                match_count += 1
    
    if total_keywords > 0 and match_count / total_keywords >= threshold:
        return 1
    
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    return 1 if similarity > threshold else 0

def calculate_answer_quality(model_answer, reference_answer):
"""
    评估回答质量：包括相关性、完整性、一致性
    返回一个综合质量分数（0-1）
    """
    # 1. 计算相似度
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    
    # 2. 检查是否包含关键信息
    important_keywords = ["法定代表人", "公司", "地址", "金额", "供应商", "采购方", "中标", "价格", "项目"]
    keyword_hit = 0
    for keyword in important_keywords:
        if keyword in reference_answer and keyword in model_answer:
            keyword_hit += 1
    
    keyword_score = keyword_hit / len(important_keywords) if important_keywords else 0
    
    # 3. 检查回答格式
    format_score = 1.0
    # 检查是否包含常见错误开头
    error_prefixes = ["对不起", "抱歉", "我不确定", "无法回答", "我不知道"]
    for prefix in error_prefixes:
        if model_answer.startswith(prefix):
            format_score -= 0.2
    
    # 4. 计算综合分数
    final_score = (similarity * 0.4) + (keyword_score * 0.4) + (format_score * 0.2)
    
    return {
        "quality_score": final_score,
        "similarity": similarity,
        "keyword_score": keyword_score,
        "format_score": format_score
    }

# ====================== 测试运行器 ======================
class LlamaModelTestRunner:
    """Llama模型测试运行器
"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = self._setup_output_dir()
        self.test_cases = load_qa_data(config.test_data_path)
        
        # 限制测试用例数量
        if len(self.test_cases) > config.max_test_cases:
            self.test_cases = self.test_cases[:config.max_test_cases]
            print(f"📊 限制测试用例数为: {len(self.test_cases)}")
        
        print(f"\n{'='*60}")
        print(f"Llama模型测试初始化完成")
        print(f"模型名称: {config.llm_config['llm_name']}")
        print(f"测试用例数: {len(self.test_cases)}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _setup_output_dir(self):
        """设置输出目录
"""
        model_name = self.config.llm_config['llm_name'].replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.output_dir, f"{model_name}_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        return output_dir
    
    def run_llama_tests(self):
        """运行Llama模型测试
"""
        print(f"\n{'='*60}")
        print(f"开始Llama模型测试")
        print(f"模型: {self.config.llm_config['llm_name']}")
        print(f"{'='*60}")
        
        # 加载Llama模型
        tokenizer, llm_model = ModelManager.load_local_models(self.config.llm_config)
        if llm_model is None:
            print(f"❌ Llama模型加载失败，测试终止")
            return
        
        # 运行测试
        scenario1_results = []
        scenario2_results = []
        
        for idx, test_case in enumerate(self.test_cases):
            question = test_case["question"]
            reference_answer = test_case["reference_answer"]
            
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
        scenario2_quality_scores = [r.get("quality_score", 0) for r in scenario2_results if "quality_score" in r]
        scenario2_answer_lengths = [r.get("answer_length", 0) for r in scenario2_results if "answer_length" in r]
        
        return {
            "test_cases_count": len(scenario1_results),
            "scenario1_avg_accuracy": np.mean(scenario1_accuracies) if scenario1_accuracies else 0,
            "scenario1_avg_quality": np.mean(scenario1_quality_scores) if scenario1_quality_scores else 0,
            "scenario1_avg_answer_length": np.mean(scenario1_answer_lengths) if scenario1_answer_lengths else 0,
            "scenario2_avg_accuracy": np.mean(scenario2_accuracies) if scenario2_accuracies else 0,
            "scenario2_avg_quality": np.mean(scenario2_quality_scores) if scenario2_quality_scores else 0,
            "scenario2_avg_answer_length": np.mean(scenario2_answer_lengths) if scenario2_answer_lengths else 0
        }
    
    def _generate_log_file(self, test_results):
        """生成日志文件
"""
        log_content = f"""Llama模型测试日志
测试时间: {test_results['test_time']}
模型名称: {test_results['llm_config']['llm_name']}
模型路径: {test_results['llm_config']['llm_local_path']}

测试配置:
  测试用例数: {test_results['test_config']['test_cases_count']}
  最大测试用例数: {test_results['test_config']['max_test_cases']}

测试结果摘要:
  场景1平均准确率: {test_results['summary']['scenario1_avg_accuracy']:.4f}
  场景1平均质量分: {test_results['summary']['scenario1_avg_quality']:.4f}
  场景1平均回答长度: {test_results['summary']['scenario1_avg_answer_length']:.2f}
  场景2平均准确率: {test_results['summary']['scenario2_avg_accuracy']:.4f}
  场景2平均质量分: {test_results['summary']['scenario2_avg_quality']:.4f}
  场景2平均回答长度: {test_results['summary']['scenario2_avg_answer_length']:.2f}

详细结果请查看results目录下的JSON文件。
"""
        
        log_file = os.path.join(self.output_dir, "logs", "test_summary.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        
        print(f"✅ 日志文件已生成: {log_file}")

# ====================== 主函数 ======================
def main():
    """主函数：运行Llama模型测试
"""
    
    print(f"{'='*60}")
    print("Llama模型测试")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有专业提示词推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建Llama测试配置
    test_config = LlamaTestConfig()
    
    # 创建测试运行器
    test_runner = LlamaModelTestRunner(test_config)
    
    # 运行Llama模型测试
    test_runner.run_llama_tests()
    
    print(f"\n{'='*60}")
    print("✅ Llama模型测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()