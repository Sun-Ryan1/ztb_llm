import torch
import json
import numpy as np
import os
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from torch.utils.tensorboard import SummaryWriter
import sys

# ---------------------- 1. 初始化output目录 ----------------------
def init_output_dir():
    output_dir = "./output_optimized"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tb_log_dir = os.path.join(output_dir, f"tb_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(tb_log_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    return output_dir, tb_log_dir, result_dir, log_dir

# ---------------------- 2. 日志重定向 ----------------------
class LoggerRedirect:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ---------------------- 3. TensorBoard初始化 ----------------------
def init_tensorboard(tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"✅ TensorBoard日志已初始化，存储路径：{tb_log_dir}")
    return writer

def log_to_tensorboard(writer, step, metrics, scenario):
    """记录TensorBoard日志，区分场景
"""
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"{metric_name}/{scenario}", metric_value, step)

# ---------------------
"""场景1：无任何提示词，直接让模型回答问题
"""inputs = tokenizer(
        question,  # 只输入问题，不加任何提示
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return model_answer, []  # 无检索文档

# ---------------------
"""场景2：使用提示词模板的直接推理，无RAG
"""prompt = f"""你是一个招投标领域的助手。请根据你的知识回答以下问题：

问题：{question}

要求：
1. 如果你知道答案，请直接给出准确简洁的回答
2. 如果不知道，请说"根据现有信息无法确定"
3. 答案要简洁准确

回答：
"""
    
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
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    model_answer = full_response.replace(prompt, "").strip()
    
    return model_answer, []  # 无检索文档

# ---------------------
"""评估回答质量：包括相关性、完整性、一致性
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

# ---------------------
"""运行双场景测试：无提示词 vs 有提示词
"""
    output_dir, tb_log_dir, result_dir, log_dir = init_output_dir()
    log_file_path = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = LoggerRedirect(log_file_path)
    tb_writer = init_tensorboard(tb_log_dir)
    
    # 修改为Qwen2.5-7B-Instruct的配置
    test_config = {
        "llm_name": "Qwen2.5-7B-Instruct",
        "llm_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen2-5-7B-Instruct",
        "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5/BAAI/bge-large-zh-v1___5"
    }
    
    # 1. 加载数据
    qa_file_path = "qa_data/520_qa.json"
    kb_file_path = "qa_data/knowledge_base.txt"
    test_cases, test_docs = load_project_qa_data(qa_file_path, kb_file_path)
    if not test_cases or not test_docs:
        print("\n❌ 无有效测试数据，测试终止")
        tb_writer.close()
        return
    
    # 2. 加载模型
    print("\n" + "="*60)
    print("加载模型...")
    tokenizer, llm_model, embedding_models = load_local_models(
        test_config["llm_name"],
        test_config["llm_local_path"],
        test_config["embedding_local_path"]
    )
    if tokenizer is None or llm_model is None or embedding_models is None:
        print("\n❌ 模型加载失败，测试终止")
        tb_writer.close()
        return
    
    # 4. 执行双场景测试
    test_results_scenario1 = []  # 场景1：无提示词
    test_results_scenario2 = []  # 场景2：有提示词
    
    print("\n" + "="*60)
    print("开始双场景模型能力测试...")
    print("场景1：无提示词直接生成")
    print("场景2：有提示词生成")
    print("="*60)
    
    for idx, case in enumerate(test_cases):
        question = case["question"]
        reference_answer = case["reference_answer"]
        relevant_docs = case["relevant_docs"]
        
        print(f"\n=== 测试用例 {idx+1}/{len(test_cases)} ===")
        print(f"问题：{question}")
        
        # 场景1：无提示词直接回答
        print("\n--- 场景1：无提示词 ---")
        try:
            model_answer1, _ = direct_inference_no_prompt(tokenizer, llm_model, question)
            accuracy1 = calculate_accuracy(model_answer1, reference_answer)
            quality_metrics1 = calculate_answer_quality(model_answer1, reference_answer)
            
            result1 = {
                "scenario": "no_prompt",
                "test_case_id": idx + 1,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer1,
                "accuracy": accuracy1,
                "answer_length": len(model_answer1),
                "quality_score": quality_metrics1["quality_score"],
                "similarity": quality_metrics1["similarity"],
                "keyword_score": quality_metrics1["keyword_score"]
            }
            
            print(f"  模型回答：{model_answer1[:80]}..." if len(model_answer1) > 80 else f"  模型回答：{model_answer1}")
            print(f"  准确率：{accuracy1:.4f} | 质量分：{quality_metrics1['quality_score']:.4f} | 长度：{len(model_answer1)}")
            
            # TensorBoard记录
            log_to_tensorboard(tb_writer, step=idx+1, metrics={
                "accuracy": accuracy1,
                "quality_score": quality_metrics1["quality_score"],
                "answer_length": len(model_answer1),
                "similarity": quality_metrics1["similarity"]
            }, scenario="scenario1_no_prompt")
            
            test_results_scenario1.append(result1)
            
        except Exception as e:
            print(f"❌ 场景1测试失败：{e}")
            test_results_scenario1.append({
                "scenario": "no_prompt",
                "test_case_id": idx + 1,
                "question": question,
                "error": str(e)
            })
        
        # 场景2：有提示词的直接回答
        print("\n--- 场景2：有提示词 ---")
        try:
            model_answer2, _ = direct_inference_with_prompt(tokenizer, llm_model, question)
            
            # 使用相关文档计算召回率
            recall2 = calculate_recall([model_answer2], relevant_docs)
            accuracy2 = calculate_accuracy(model_answer2, reference_answer)
            quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
            
            result2 = {
                "scenario": "with_prompt",
                "test_case_id": idx + 1,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer2,
                "relevant_docs": relevant_docs,
                "recall_score": recall2,
                "accuracy": accuracy2,
                "answer_length": len(model_answer2),
                "quality_score": quality_metrics2["quality_score"],
                "similarity": quality_metrics2["similarity"],
                "keyword_score": quality_metrics2["keyword_score"]
            }
            
            print(f"  模型回答：{model_answer2[:80]}..." if len(model_answer2) > 80 else f"  模型回答：{model_answer2}")
            print(f"  召回率：{recall2:.4f} | 准确率：{accuracy2:.4f} | 质量分：{quality_metrics2['quality_score']:.4f}")
            
            # TensorBoard记录
            log_to_tensorboard(tb_writer, step=idx+1, metrics={
                "accuracy": accuracy2,
                "recall_score": recall2,
                "quality_score": quality_metrics2["quality_score"],
                "answer_length": len(model_answer2),
                "similarity": quality_metrics2["similarity"]
            }, scenario="scenario2_with_prompt")
            
            test_results_scenario2.append(result2)
            
        except Exception as e:
            print(f"❌ 场景2测试失败：{e}")
            test_results_scenario2.append({
                "scenario": "with_prompt",
                "test_case_id": idx + 1,
                "question": question,
                "error": str(e)
            })
        
        # 进度报告
        if (idx + 1) % 5 == 0:
            if test_results_scenario1:
                avg_acc1 = sum([r.get("accuracy", 0) for r in test_results_scenario1 if "accuracy" in r]) / len([r for r in test_results_scenario1 if "accuracy" in r])
                print(f"\n📊 当前进度：{idx+1}/{len(test_cases)}")
                print(f"  场景1平均准确率：{avg_acc1:.4f}")
            
            if test_results_scenario2 and any("accuracy" in r for r in test_results_scenario2):
                valid_results2 = [r for r in test_results_scenario2 if "accuracy" in r]
                if valid_results2:
                    avg_acc2 = sum([r["accuracy"] for r in valid_results2]) / len(valid_results2)
                    avg_recall2 = sum([r.get("recall_score", 0) for r in valid_results2]) / len(valid_results2)
                    print(f"  场景2平均准确率：{avg_acc2:.4f}，平均召回率：{avg_recall2:.4f}")
    
    # 5. 保存测试结果
    # 场景1结果
    result_file_name1 = f"{test_config['llm_name']}_scenario1_no_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file_path1 = os.path.join(result_dir, result_file_name1)
    with open(result_file_path1, "w", encoding="utf-8") as f:
        json.dump(test_results_scenario1, f, ensure_ascii=False, indent=2)
    
    # 场景2结果
    result_file_name2 = f"{test_config['llm_name']}_scenario2_with_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file_path2 = os.path.join(result_dir, result_file_name2)
    with open(result_file_path2, "w", encoding="utf-8") as f:
        json.dump(test_results_scenario2, f, ensure_ascii=False, indent=2)
    
    # 对比分析结果
    comparison_file_name = f"{test_config['llm_name']}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison_file_path = os.path.join(result_dir, comparison_file_name)
    
    # 6. 测试总结与对比分析
    print("\n" + "="*60)
    print("✅ 双场景测试完成！")
    print(f"📊 场景1结果：{result_file_path1}")
    print(f"📊 场景2结果：{result_file_path2}")
    print(f"📈 TensorBoard日志：{tb_log_dir}")
    print(f"📝 运行日志：{log_file_path}")
    
    # 计算统计指标
    if test_results_scenario1 and any("accuracy" in r for r in test_results_scenario1):
        valid_scenario1 = [r for r in test_results_scenario1 if "accuracy" in r]
        total_accuracy1 = sum([r["accuracy"] for r in valid_scenario1]) / len(valid_scenario1)
        avg_length1 = sum([r["answer_length"] for r in valid_scenario1]) / len(valid_scenario1)
        avg_quality1 = sum([r.get("quality_score", 0) for r in valid_scenario1]) / len(valid_scenario1)
        
        print(f"\n📊 场景1统计（无提示词）：")
        print(f"  平均准确率：{total_accuracy1:.4f}")
        print(f"  平均质量分：{avg_quality1:.4f}")
        print(f"  平均回答长度：{avg_length1:.1f}")
        print(f"  有效测试用例：{len(valid_scenario1)}/{len(test_cases)}")
    
    if test_results_scenario2 and any("accuracy" in r for r in test_results_scenario2):
        valid_scenario2 = [r for r in test_results_scenario2 if "accuracy" in r]
        total_accuracy2 = sum([r["accuracy"] for r in valid_scenario2]) / len(valid_scenario2)
        avg_recall2 = sum([r.get("recall_score", 0) for r in valid_scenario2]) / len(valid_scenario2)
        avg_length2 = sum([r["answer_length"] for r in valid_scenario2]) / len(valid_scenario2)
        avg_quality2 = sum([r.get("quality_score", 0) for r in valid_scenario2]) / len(valid_scenario2)
        
        print(f"\n📊 场景2统计（有提示词）：")
        print(f"  平均准确率：{total_accuracy2:.4f}")
        print(f"  平均召回率：{avg_recall2:.4f}")
        print(f"  平均质量分：{avg_quality2:.4f}")
        print(f"  平均回答长度：{avg_length2:.1f}")
        print(f"  有效测试用例：{len(valid_scenario2)}/{len(test_cases)}")
    
    # 对比分析
    if test_results_scenario1 and test_results_scenario2:
        print(f"\n📊 场景对比分析：")
        
        # 计算准确率提升
        if total_accuracy1 > 0 and total_accuracy2 > 0:
            accuracy_improvement = ((total_accuracy2 - total_accuracy1) / total_accuracy1) * 100
            print(f"  准确率提升：{accuracy_improvement:+.2f}%")
        
        # 计算召回率提升
        if 'avg_recall2' in locals():
            recall_improvement = avg_recall2 - 0  # 场景1无检索，召回率为0
            print(f"  召回率提升：{recall_improvement:+.2f}")
        
        # 计算质量分提升
        if avg_quality1 > 0 and avg_quality2 > 0:
            quality_improvement = ((avg_quality2 - avg_quality1) / avg_quality1) * 100
            print(f"  质量分提升：{quality_improvement:+.2f}%")
        
        # 计算回答长度差异
        length_difference = avg_length2 - avg_length1
        print(f"  回答长度差异：{length_difference:+.1f} 字符")
        
        # 保存对比分析结果
        comparison_results = {
            "model": test_config["llm_name"],
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(test_cases),
            "scenario1_stats": {
                "avg_accuracy": total_accuracy1 if 'total_accuracy1' in locals() else 0,
                "avg_quality": avg_quality1 if 'avg_quality1' in locals() else 0,
                "avg_length": avg_length1 if 'avg_length1' in locals() else 0,
                "valid_cases": len(valid_scenario1) if 'valid_scenario1' in locals() else 0
            },
            "scenario2_stats": {
                "avg_accuracy": total_accuracy2 if 'total_accuracy2' in locals() else 0,
                "avg_recall": avg_recall2 if 'avg_recall2' in locals() else 0,
                "avg_quality": avg_quality2 if 'avg_quality2' in locals() else 0,
                "avg_length": avg_length2 if 'avg_length2' in locals() else 0,
                "valid_cases": len(valid_scenario2) if 'valid_scenario2' in locals() else 0
            },
            "improvements": {
                "accuracy_improvement": accuracy_improvement if 'accuracy_improvement' in locals() else 0,
                "recall_improvement": recall_improvement if 'recall_improvement' in locals() else 0,
                "quality_improvement": quality_improvement if 'quality_improvement' in locals() else 0,
                "length_difference": length_difference if 'length_difference' in locals() else 0
            }
        }
        
        with open(comparison_file_path, "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 对比分析结果已保存：{comparison_file_path}")
    
    print("="*60)
    
    tb_writer.close()

# ---------------------- 14. 一键运行 ----------------------
if __name__ == "__main__":
    print("="*60)
    print("双场景大模型能力调研测试")
    print("模型：Qwen2.5-7B-Instruct + BGE-large-zh-v1.5")
    print("场景1：无提示词直接生成（测试原始模型能力）")
    print("场景2：有提示词生成（测试提示词优化效果）")
    print("="*60)
    
    run_dual_scenario_test()
