import torch
import json
import numpy as np
import os
import re
import sys
import gc
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from difflib import SequenceMatcher

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类
"""
    def __init__(self, llm_name, llm_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path

# ====================== InternLM2模型配置 ======================
class ILM2TestConfig:
    """InternLM2模型测试配置
"""
    def __init__(self):
        # InternLM2模型配置
        self.llm_config = {
            "llm_name": "internlm2_5-7b-chat",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/Shanghai_AI_Laboratory/internlm2_5-7b-chat"
        }
        
        # 测试数据路径
        self.test_data_path = "qa_data/100_qa.json"
        
        # 基础输出目录
        self.base_output_dir = "./ilm2_model_results_noRAG"
        
        # 测试参数
        self.max_test_cases = 50
        self.batch_size = 16
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.75

# ====================== 模型管理器 ======================
class ModelManager:
    """模型管理器（针对InternLM2优化）
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
        """加载本地模型（针对InternLM2优化）
        返回: (tokenizer, llm_model)
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
            return tokenizer, llm_model
        except Exception as e:
            print(f"❌ 大模型加载失败，错误信息：{e}")
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

# ====================== InternLM2专用生成函数 ======================
def safe_generate_internlm2_response(tokenizer, model, prompt, max_new_tokens=200):
    """InternLM2专用生成函数
"""
    try:
        # 清理prompt中的特殊字符
        prompt = clean_special_characters(prompt)
        
        # 编码输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True
        )
        
        # 直接移动整个inputs到设备，保持内部一致性
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 确保有attention_mask
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # 验证维度一致性
        if inputs['input_ids'].shape[1] != inputs['attention_mask'].shape[1]:
            print(f"⚠️ 维度不匹配: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")
            # 使用较短的维度
            min_len = min(inputs['input_ids'].shape[1], inputs['attention_mask'].shape[1])
            inputs['input_ids'] = inputs['input_ids'][:, :min_len]
            inputs['attention_mask'] = inputs['attention_mask'][:, :min_len]
        
        # 生成参数
        generate_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "max_new_tokens": min(max_new_tokens, 256),
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "use_cache": False,
        }
        
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
        
        # 解码并清理
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        elif prompt in response:
            response = response.split(prompt)[-1].strip()
        
        return response if response else "未知"
    
    except RuntimeError as e:
        if "dimension" in str(e).lower() or "size" in str(e).lower():
            # 维度错误，尝试简化方法
            print(f"⚠️ 维度错误，使用简化方法: {str(e)[:100]}")
            return safe_generate_simple(tokenizer, model, prompt, max_new_tokens)
        else:
            print(f"❌ InternLM2生成失败: {str(e)[:200]}")
            return "生成失败"
    except Exception as e:
        print(f"❌ InternLM2生成失败: {str(e)[:200]}")
        return "生成失败"

def safe_generate_simple(tokenizer, model, prompt, max_new_tokens=200):
    """简化生成方法，避免复杂参数
"""
    try:
        # 超简化的prompt
        simple_prompt = f"请回答：{prompt[:200]}"
        
        # 最简单的方式：只使用input_ids，不使用attention_mask
        input_ids = tokenizer.encode(simple_prompt, return_tensors="pt", 
                                    truncation=True, max_length=256)
        input_ids = input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=min(max_new_tokens, 200),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                use_cache=False,
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入
        if response.startswith(simple_prompt):
            response = response[len(simple_prompt):].strip()
        elif simple_prompt in response:
            response = response.split(simple_prompt)[-1].strip()
        
        return response if response else "未知"
    
    except Exception as e:
        print(f"❌ 简化生成失败: {str(e)[:200]}")
        return "生成失败"

def clean_special_characters(text):
    """清理文本中的特殊字符
"""
    if not text:
        return ""
    
    # 替换点号中间的空格
    text = text.replace("沈.阳", "沈阳")
    
    # 移除其他特殊字符但保留中文标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：\"\'（）《》、]', '', text)
    
    return text

# ====================== 推理函数 ======================
def direct_inference_no_prompt(tokenizer, llm_model, question):
    """场景1：无任何提示词，直接让模型回答问题
"""
    # 清理问题中的特殊字符
    question = clean_special_characters(question)
    
    # 无提示词，直接使用问题作为输入
    prompt = question
    
    model_answer = safe_generate_internlm2_response(tokenizer, llm_model, prompt, max_new_tokens=200)
    
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

# 示例
## 示例1
问：多次招标都是同一供应商满足参数要求，可变为单一来源吗？
答：公开招标过程中提交投标文件或者经评审实质性响应招标文件要求的供应商只有一家时，可以申请单一来源采购方式。具体标准可参考《中央预算单位变更政府采购方式审批管理办法》（财库〔2015〕36 号）第十条规定或者本地方一些规范性文件规定。

## 示例2
问：中标公告发出后发现第一名为无效投标时，招标人应如何处理？
答：由招标人依据中标条件从其余投标人中重新确定中标人或者依照招投标法重新进行招标。

## 示例3
问：招标文件要求中标人提交履约保证金的最高限额是多少？
答：履约保证金不得超过中标合同金额的10%。

# 现在请回答以下问题
问：{question}
答："""model_answer = safe_generate_internlm2_response(tokenizer, llm_model, prompt, max_new_tokens=300)
    
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
    
    # 清理答案
    model_answer_clean = model_answer.strip()
    reference_answer_clean = reference_answer.strip()
    
    if reference_answer_clean in model_answer_clean or model_answer_clean in reference_answer_clean:
        return 1
    
    important_keywords = ["法定代表人", "公司", "地址", "金额", "供应商", "采购方", "中标", "价格", "项目", "手术室", "传递窗"]
    match_count = 0
    total_keywords = 0
    
    for keyword in important_keywords:
        if keyword in reference_answer_clean:
            total_keywords += 1
            if keyword in model_answer_clean:
                match_count += 1
    
    if total_keywords > 0 and match_count / total_keywords >= threshold:
        return 1
    
    ref_entities = extract_entities_from_text(reference_answer_clean)
    model_entities = extract_entities_from_text(model_answer_clean)
    
    if ref_entities:
        common_entities = set(ref_entities) & set(model_entities)
        if len(common_entities) / len(ref_entities) >= 0.5:
            return 1
    
    similarity = SequenceMatcher(None, model_answer_clean, reference_answer_clean).ratio()
    return 1 if similarity > threshold else 0

def extract_entities_from_text(text):
"""从文本中提取实体"""entities = []
    
    company_patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)',
        r'供应商[：:]?([\u4e00-\u9fa5a-zA-Z0-9]{2,})',
        r'由([\u4e00-\u9fa5a-zA-Z0-9]{2,})提供'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    product_indicators = ["产品", "设备", "仪器", "系统", "项目", "服务"]
    words = text.split()
    for word in words:
        if any(indicator in word for indicator in product_indicators):
            entities.append(word)
    
    price_matches = re.findall(r'(\d+\.?\d*)元', text)
    entities.extend(price_matches)
    
    return list(set(entities))

def calculate_answer_quality(model_answer, reference_answer):
"""
    评估回答质量：包括相关性、完整性、一致性
    返回一个综合质量分数（0-1）
    """
    # 1. 计算相似度
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    
    # 2. 检查是否包含关键信息
    important_keywords = ["法定代表人", "公司", "地址", "金额", "供应商", "采购方", "中标", "价格", "项目", "手术室", "传递窗"]
    keyword_hit = 0
    for keyword in important_keywords:
        if keyword in reference_answer and keyword in model_answer:
            keyword_hit += 1
    
    keyword_score = keyword_hit / len(important_keywords) if important_keywords else 0
    
    # 3. 检查回答格式
    format_score = 1.0
    # 检查是否包含常见错误开头
    error_prefixes = ["对不起", "抱歉", "我不确定", "无法回答", "我不知道", "生成失败"]
    for prefix in error_prefixes:
        if model_answer.startswith(prefix):
            format_score -= 0.3
    
    # 4. 计算综合分数
    final_score = (similarity * 0.4) + (keyword_score * 0.4) + (format_score * 0.2)
    
    return {
        "quality_score": final_score,
        "similarity": similarity,
        "keyword_score": keyword_score,
        "format_score": format_score
    }

# ====================== 日志重定向类 ======================
class LoggerRedirect:
    """日志重定向类，将控制台输出同时写入文件
"""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()

# ====================== 测试运行器 ======================
class ILM2ModelTestRunner:
    """InternLM2模型测试运行器
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
        print(f"{config.llm_config['llm_name']} 模型测试框架初始化完成")
        print(f"测试用例数: {len(self.test_cases)}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _setup_output_dir(self):
        """设置输出目录，包含模型名称
"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建模型专属的输出目录
        model_output_dir = os.path.join(
            self.config.base_output_dir, 
            f"{self.config.llm_config['llm_name']}_{timestamp}"
        )
        
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(os.path.join(model_output_dir, "results"), exist_ok=True)
        
        return model_output_dir
    
    def run_ilm2_tests(self):
        """运行InternLM2模型测试
"""
        print(f"\n{'='*60}")
        print(f"开始 {self.config.llm_config['llm_name']} 模型测试")
        print(f"测试场景：1.无提示词 2.有专业提示词")
        print(f"注意：不使用RAG，不进行文档检索")
        print(f"{'='*60}")
        
        # 加载InternLM2模型
        tokenizer, llm_model = ModelManager.load_local_models(self.config.llm_config)
        if llm_model is None:
            print(f"❌ {self.config.llm_config['llm_name']} 模型加载失败，跳过测试")
            return
        
        # 运行测试
        scenario1_results = []
        scenario2_results = []
        
        for idx, test_case in enumerate(self.test_cases):
            question = test_case["question"]
            reference_answer = test_case["reference_answer"]
            
            print(f"\n=== 测试用例 {idx+1}/{len(self.test_cases)} ===")
            print(f"问题：{question[:100]}..." if len(question) > 100 else f"问题：{question}")
            
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
    
    def _generate_brief_statistics(self, llm_results):
        """生成简要统计报告
"""
        stats_file = os.path.join(self.output_dir, f"{self.config.llm_config['llm_name']}_brief_statistics.txt")
        
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f"{self.config.llm_config['llm_name']} 模型测试简要统计\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"模型名称: {llm_results['model_name']}\n")
            f.write(f"测试时间: {llm_results['test_time']}\n")
            f.write(f"测试用例数: {llm_results['test_cases_count']}\n\n")
            
            f.write("📊 测试结果摘要:\n")
            f.write(f"   场景1平均准确率: {llm_results['summary']['scenario1_avg_accuracy']:.4f}\n")
            f.write(f"   场景1平均质量分: {llm_results['summary']['scenario1_avg_quality']:.4f}\n")
            f.write(f"   场景1平均回答长度: {llm_results['summary']['scenario1_avg_answer_length']:.1f}\n\n")
            
            f.write(f"   场景2平均准确率: {llm_results['summary']['scenario2_avg_accuracy']:.4f}\n")
            f.write(f"   场景2平均质量分: {llm_results['summary']['scenario2_avg_quality']:.4f}\n")
            f.write(f"   场景2平均回答长度: {llm_results['summary']['scenario2_avg_answer_length']:.1f}\n\n")
            
            # 成功/失败统计
            scenario1_success = sum(1 for r in llm_results['scenario1_results'] if r.get("accuracy", 0) == 1)
            scenario1_fail = len(llm_results['scenario1_results'])
"""主函数：运行InternLM2模型测试
"""
    
    print(f"{'='*60}")
    print("模型选型第一阶段测试框架")
    print("目标：纯模型能力测试，不使用RAG")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有专业提示词推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建测试配置
    test_config = ILM2TestConfig()
    
    # 创建测试运行器
    test_runner = ILM2ModelTestRunner(test_config)
    
    # 运行该模型的测试
    test_runner.run_ilm2_tests()
    
    print(f"\n{'='*60}")
    print("✅ 模型测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()