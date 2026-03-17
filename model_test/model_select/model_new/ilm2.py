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
    """模型配置类"""
    def __init__(self, llm_name, llm_local_path, embedding_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path
        self.embedding_local_path = embedding_local_path

# ====================== 测试配置 ======================
class TestConfig:
    """测试配置类 - 第一阶段大模型选型"""
    def __init__(self):
        # 候选大模型配置
        self.llm_candidates = [
            {
                "llm_name": "internlm2_5-7b-chat",
                "llm_local_path": "/mnt/workspace/data/modelscope/cache/Shanghai_AI_Laboratory/internlm2_5-7b-chat",
                "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5/BAAI/bge-large-zh-v1___5"
            },
        ]
        
        # 测试数据路径
        self.test_data_path = "qa_data/100_qa.json"
        self.knowledge_base_path = "qa_data/knowledge_base.txt"
        
        # 基础输出目录
        self.base_output_dir = "./ilm2_model_selection_results"
        
        # 测试参数
        self.max_test_cases = 50
        self.batch_size = 16
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.75

# ====================== 模型管理器 ======================
class ModelManager:
    """模型管理器（针对InternLM2优化）"""
    
    @staticmethod
    def get_bnb_config():
        """获取量化参数配置"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    @staticmethod
    def load_local_models(config):
        """
        加载本地模型（针对InternLM2优化）
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
        
        # 3. 加载Embedding模型（固定使用BGE-large-zh-v1.5）
        print(f"正在加载本地BGE Embedding模型")
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                local_files_only=True
            )
            embedding_model = AutoModel.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            embedding_model.eval()
            print(f"✅ BGE Embedding模型 加载成功")
        except Exception as e:
            print(f"❌ BGE Embedding模型加载失败，错误信息：{e}")
            return None, None, None
        
        return tokenizer, llm_model, (embedding_tokenizer, embedding_model)
    
    @staticmethod
    def cleanup_models(llm_model, embedding_models=None):
        """清理模型资源"""
        if llm_model is not None:
            del llm_model
        
        if embedding_models is not None:
            embedding_tokenizer, embedding_model = embedding_models
            del embedding_tokenizer
            del embedding_model
        
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ 模型资源已清理")

# ====================== InternLM2专用生成函数（修复版本）======================
def safe_generate_internlm2_response(tokenizer, model, prompt, max_new_tokens=200):
    """
    InternLM2专用生成函数，修复张量维度不匹配问题
    """
    try:
        # 清理prompt中的特殊字符
        prompt = clean_special_characters(prompt)
        
        # 编码输入，使用更安全的参数设置（参考图片中的成功做法）
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # 进一步减少最大长度
            padding=False,    # 关键修复：关闭padding，避免生成额外维度
            return_attention_mask=True
        )
        
        # 直接移动整个inputs到设备，保持内部一致性
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 确保有attention_mask，如果没有则创建
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # 验证维度一致性
        if inputs['input_ids'].shape[1] != inputs['attention_mask'].shape[1]:
            print(f"⚠️ 维度不匹配: input_ids={inputs['input_ids'].shape}, attention_mask={inputs['attention_mask'].shape}")
            # 使用较短的维度
            min_len = min(inputs['input_ids'].shape[1], inputs['attention_mask'].shape[1])
            inputs['input_ids'] = inputs['input_ids'][:, :min_len]
            inputs['attention_mask'] = inputs['attention_mask'][:, :min_len]
        
        # 生成参数（使用更保守的设置）
        generate_kwargs = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            "max_new_tokens": min(max_new_tokens, 256),  # 进一步限制生成长度
            "do_sample": False,  # 使用贪婪解码
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,  # 降低重复惩罚
            "no_repeat_ngram_size": 3,   # 减少ngram大小
            "use_cache": False,  # 关键修复：关闭cache避免维度问题
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
    """简化生成方法，避免复杂参数"""
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
                no_repeat_ngram_size=3,  # 完全禁用ngram限制
                use_cache=False,  # 关闭cache
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
    """清理文本中的特殊字符"""
    if not text:
        return ""
    
    # 替换点号中间的空格
    text = text.replace("沈.阳", "沈阳")
    
    # 移除其他特殊字符但保留中文标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；："\'（）《》、]', '', text)
    
    return text

# ====================== BGE文本转向量函数 ======================
def bge_embedding_encode(embedding_models, text, batch_mode=False):
    """BGE文本转向量函数（保持原有实现）"""
    embedding_tokenizer, embedding_model = embedding_models
    text = str(text).strip()
    if not text:
        return np.array([])
    
    max_length = 512
    
    if batch_mode:
        inputs = embedding_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(embedding_model.device)
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        return embeddings.cpu().numpy()
    else:
        inputs = embedding_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(embedding_model.device)
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            hidden_state = outputs.last_hidden_state[:, 0]
            vec = hidden_state.cpu().numpy().squeeze()
        
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

# ====================== 构建FAISS向量索引 ======================
def build_vector_index(embedding_models, docs, batch_size=32):
    """构建FAISS向量索引（保持原有实现）"""
    doc_vectors = []
    valid_docs = []
    
    print(f"正在编码 {len(docs)} 条文档...")
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_vectors = bge_embedding_encode(embedding_models, batch_docs, batch_mode=True)
        
        for j, vec in enumerate(batch_vectors):
            if vec.size > 0:
                doc_vectors.append(vec)
                valid_docs.append(batch_docs[j])
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已编码 {min(i+batch_size, len(docs))}/{len(docs)} 条文档")
    
    if not doc_vectors:
        print("❌ 无有效文档向量，无法构建FAISS索引")
        return None, None
    
    doc_vectors = np.array(doc_vectors, dtype=np.float32)
    vector_dim = doc_vectors.shape[1]
    
    index = faiss.IndexFlatIP(vector_dim)
    index.add(doc_vectors)
    
    print(f"✅ 共构建 {len(valid_docs)} 条有效文档的FAISS索引")
    print(f"  向量维度: {vector_dim}")
    print(f"  索引类型: IndexFlatIP (内积相似度)")
    
    return index, valid_docs

# ====================== 检索函数 ======================
def enhanced_retrieval(embedding_models, index, docs, question, top_k=5, similarity_threshold=0.75):
    """改进的检索函数（保持原有实现）"""
    results = []
    
    # 1. 向量检索
    question_vector = bge_embedding_encode(embedding_models, question)
    if question_vector.size == 0:
        return []
    
    distances, indices = index.search(question_vector.reshape(1, -1), top_k * 5)
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(docs) and dist > similarity_threshold:
            results.append({
                "doc": docs[idx],
                "similarity": float(dist),
                "type": "vector",
                "index": idx
            })
    
    # 2. 关键词检索（后备）
    if len(results) < 3:
        keyword_results = keyword_based_retrieval(question, docs)
        results.extend(keyword_results)
    
    # 3. 去重和排序
    unique_results = []
    seen_indices = set()
    
    for result in results:
        idx = result.get("index", -1)
        if idx >= 0 and idx not in seen_indices:
            seen_indices.add(idx)
            unique_results.append(result)
    
    unique_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return [r["doc"] for r in unique_results[:top_k]]

def keyword_based_retrieval(query, docs):
    """关键词检索（后备）"""
    results = []
    
    companies = extract_company_names(query)
    keywords = extract_keywords(query)
    
    for idx, doc in enumerate(docs):
        score = 0
        
        for company in companies:
            if company in doc:
                score += 1.0
                break
        
        for keyword in keywords:
            if keyword in doc:
                score += 0.5
        
        if score > 0:
            results.append({
                "doc": doc,
                "similarity": min(score / 2.0, 1.0),
                "type": "keyword",
                "index": idx
            })
    
    return results

def extract_company_names(text):
    """提取公司名"""
    patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)',
        r'(?:关于|咨询|查询)([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:的|信息)?'
    ]
    
    companies = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                companies.extend([m for m in match if m and len(m) > 1])
            elif isinstance(match, str):
                if match and len(match) > 1:
                    companies.append(match)
    
    return list(set(companies))

def extract_keywords(text):
    """提取关键词"""
    keywords = []
    
    bid_keywords = [
        "法定代表人", "法人代表", "法人", "负责人",
        "供应商", "供货商", "供应方", "承包商",
        "采购方", "买方", "购买方", "需求方",
        "招标", "中标", "投标", "项目",
        "合同金额", "合同价", "成交金额", "中标金额",
        "地址", "注册地址", "经营地址",
        "手术室", "传递窗", "使用安全", "便捷", "双向防护", "用材精良"  # 添加特定关键词
    ]
    
    for keyword in bid_keywords:
        if keyword in text:
            keywords.append(keyword)
    
    return keywords

# ====================== 推理函数 ======================
def direct_inference_no_prompt(tokenizer, llm_model, question):
    """场景1：无任何提示词，直接让模型回答问题"""
    # 清理问题中的特殊字符
    question = clean_special_characters(question)
    
    # 无提示词，直接使用问题作为输入
    # 这是为了测试模型的基础学习能力
    prompt = question
    
    model_answer = safe_generate_internlm2_response(tokenizer, llm_model, prompt, max_new_tokens=200)
    
    return model_answer, []  # 无检索文档

def optimized_rag_inference(tokenizer, llm_model, embedding_models, index, docs, question):
    """场景2：使用提示词模板的RAG推理"""
    retrieved_docs = enhanced_retrieval(embedding_models, index, docs, question, top_k=5, similarity_threshold=0.75)
    
    if not retrieved_docs:
        return "根据现有信息无法确定。", []
    
    # 清理检索到的文档
    cleaned_docs = []
    for doc in retrieved_docs:
        cleaned_doc = clean_special_characters(doc)
        if cleaned_doc:
            cleaned_docs.append(cleaned_doc)
    
    if not cleaned_docs:
        return "根据现有信息无法确定。", []
    
    # 构建上下文信息
    context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(cleaned_docs[:3])])  # 限制文档数量
    
    # 使用专业提示词模板（含上下文）
    prompt = f"""# 角色定位
你是聚焦招投标采购全流程的专业智能问答系统，需严格依据《招标投标法》《政府采购法》等法规，精准解答政策合规、业务操作、物资产品、电子系统操作等领域问题。

# 回答要求
1. 准确性：严格依据相关法规和政策，确保信息准确无误
2. 完整性：全面覆盖问题要点，提供详细的分析和解释
3. 专业性：正确使用专业术语，体现专业知识和分析能力
4. 清晰性：语言流畅，逻辑清晰，结构合理

# 参考信息
{context}

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
答："""
    
    model_answer = safe_generate_internlm2_response(tokenizer, llm_model, prompt, max_new_tokens=200)
    
    return model_answer, retrieved_docs

# ====================== 数据加载器 ======================
def load_qa_data(qa_file_path="qa_data/100_qa.json", kb_file_path="qa_data/knowledge_base.txt"):
    """加载QA数据（保持原有实现）"""
    try:
        # 1. 加载知识库文档
        if os.path.exists(kb_file_path):
            with open(kb_file_path, "r", encoding="utf-8") as f:
                knowledge_docs = [line.strip() for line in f if line.strip()]
            print(f"✅ 加载 {len(knowledge_docs)} 条知识库文档")
        else:
            print("⚠️  未找到知识库文件，将从问答对中构建")
            knowledge_docs = []
        
        # 2. 加载问答对
        with open(qa_file_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        test_cases = []
        for idx, item in enumerate(qa_data):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            
            if not question or not answer:
                continue
            
            # 获取相关文档
            relevant_docs = []
            
            if "relevant_documents" in item:
                relevant_docs = item["relevant_documents"]
            elif "relevant_doc_indices" in item and knowledge_docs:
                for doc_idx in item["relevant_doc_indices"]:
                    if doc_idx < len(knowledge_docs):
                        relevant_docs.append(knowledge_docs[doc_idx])
            
            # 如果没有找到相关文档，使用答案作为备选
            if not relevant_docs:
                relevant_docs = [answer]
            
            test_cases.append({
                "question": question,
                "reference_answer": answer,
                "relevant_docs": relevant_docs,
                "scene": item.get("scene", "unknown"),
                "source_table": item.get("source_table", "unknown")
            })
        
        print(f"✅ 加载 {len(test_cases)} 条有效测试用例")
        
        # 3. 合并知识库（如果知识库为空）
        if not knowledge_docs:
            all_docs_set = set()
            for case in test_cases:
                for doc in case["relevant_docs"]:
                    all_docs_set.add(doc)
            knowledge_docs = list(all_docs_set)
            print(f"📚 从问答对构建 {len(knowledge_docs)} 条知识库文档")
        
        return test_cases, knowledge_docs
    
    except FileNotFoundError:
        print(f"❌ 未找到文件：{qa_file_path}")
        return [], []
    except Exception as e:
        print(f"❌ 加载问答对失败：{e}")
        return [], []

# ====================== 评估函数 ======================
def calculate_recall(retrieved_docs, relevant_docs):
    """计算召回率"""
    if not retrieved_docs or not relevant_docs:
        return 0
    
    for retrieved_doc in retrieved_docs:
        for relevant_doc in relevant_docs:
            if is_doc_related(retrieved_doc, relevant_doc):
                return 1
    
    return 0

def is_doc_related(doc1, doc2):
    """检查文档是否相关"""
    entities1 = extract_entities_from_text(doc1)
    entities2 = extract_entities_from_text(doc2)
    
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, doc1, doc2).ratio()
    return similarity > 0.6

def extract_entities_from_text(text):
    """从文本中提取实体"""
    entities = []
    
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

def calculate_accuracy(model_answer, reference_answer, threshold=0.6):
    """计算准确率"""
    if not model_answer or not reference_answer:
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
    
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, model_answer_clean, reference_answer_clean).ratio()
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
    """日志重定向类，将控制台输出同时写入文件"""
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
class ModelTestRunner:
    """模型测试运行器"""
    
    def __init__(self, config, model_name, model_config):
        self.config = config
        self.model_name = model_name
        self.model_config = model_config
        self.output_dir = self._setup_output_dir()
        
        # 设置日志重定向
        log_file_path = os.path.join(self.output_dir, "test_log.txt")
        sys.stdout = LoggerRedirect(log_file_path)
        
        self.test_cases, self.knowledge_docs = load_qa_data(
            config.test_data_path, 
            config.knowledge_base_path
        )
        
        # 限制测试用例数量
        if len(self.test_cases) > config.max_test_cases:
            self.test_cases = self.test_cases[:config.max_test_cases]
            print(f"📊 限制测试用例数为: {len(self.test_cases)}")
        
        print(f"\n{'='*60}")
        print(f"{self.model_name} 模型测试框架初始化完成")
        print(f"测试用例数: {len(self.test_cases)}")
        print(f"知识库文档数: {len(self.knowledge_docs)}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _setup_output_dir(self):
        """设置输出目录，包含模型名称"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建模型专属的输出目录
        model_output_dir = os.path.join(
            self.config.base_output_dir, 
            f"{self.model_name}_{timestamp}"
        )
        
        os.makedirs(model_output_dir, exist_ok=True)
        os.makedirs(os.path.join(model_output_dir, "results"), exist_ok=True)
        
        return model_output_dir
    
    def run_llm_selection_tests(self):
        """运行LLM测试"""
        print(f"\n{'='*60}")
        print(f"开始 {self.model_name} 模型测试")
        print(f"使用固定Embedding模型: BGE-large-zh-v1.5")
        print(f"使用固定向量库: FAISS")
        print(f"{'='*60}")
        
        # 加载模型
        tokenizer, llm_model, embedding_models = ModelManager.load_local_models(self.model_config)
        if llm_model is None:
            print(f"❌ {self.model_name} 模型加载失败，跳过测试")
            return
        
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
            print(f"问题：{question[:100]}..." if len(question) > 100 else f"问题：{question}")
            
            # 场景1：无提示词直接回答
            print("\n--- 场景1：无提示词直接回答（测试学习能力）---")
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
                
                scenario1_results.append(result1)
                
            except Exception as e:
                print(f"❌ 场景1测试失败：{str(e)[:200]}")
                scenario1_results.append({
                    "scenario": "no_prompt",
                    "test_case_id": idx + 1,
                    "question": question,
                    "error": str(e)[:200]
                })
            
            # 场景2：有提示词的RAG回答
            print("\n--- 场景2：有提示词RAG回答（测试可训练能力）---")
            if index is not None:
                try:
                    model_answer2, retrieved_docs = optimized_rag_inference(
                        tokenizer, llm_model, embedding_models, index, enhanced_docs, question
                    )
                    
                    recall2 = calculate_recall(retrieved_docs, relevant_docs)
                    accuracy2 = calculate_accuracy(model_answer2, reference_answer)
                    quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
                    
                    result2 = {
                        "scenario": "with_prompt_rag",
                        "test_case_id": idx + 1,
                        "question": question,
                        "reference_answer": reference_answer,
                        "model_answer": model_answer2,
                        "retrieved_docs": retrieved_docs[:3] if retrieved_docs else [],
                        "relevant_docs": relevant_docs,
                        "recall_score": recall2,
                        "accuracy": accuracy2,
                        "answer_length": len(model_answer2),
                        "quality_score": quality_metrics2["quality_score"],
                        "similarity": quality_metrics2["similarity"],
                        "keyword_score": quality_metrics2["keyword_score"],
                        "retrieved_count": len(retrieved_docs),
                        "relevant_count": len(relevant_docs)
                    }
                    
                    print(f"  模型回答：{model_answer2[:80]}..." if len(model_answer2) > 80 else f"  模型回答：{model_answer2}")
                    print(f"  召回率：{recall2:.4f} | 准确率：{accuracy2:.4f} | 质量分：{quality_metrics2['quality_score']:.4f}")
                    print(f"  检索文档数：{len(retrieved_docs)} | 相关文档数：{len(relevant_docs)}")
                    
                    if retrieved_docs:
                        print(f"  首条检索文档：{retrieved_docs[0][:60]}...")
                    
                    scenario2_results.append(result2)
                    
                except Exception as e:
                    print(f"❌ 场景2测试失败：{str(e)[:200]}")
                    scenario2_results.append({
                        "scenario": "with_prompt_rag",
                        "test_case_id": idx + 1,
                        "question": question,
                        "error": str(e)[:200]
                    })
            else:
                print("❌ 场景2跳过：FAISS索引未构建")
                scenario2_results.append({
                    "scenario": "with_prompt_rag",
                    "test_case_id": idx + 1,
                    "question": question,
                    "error": "FAISS索引未构建"
                })
            
            # 进度报告
            if (idx + 1) % 5 == 0:
                if scenario1_results:
                    valid_scenario1 = [r for r in scenario1_results if "accuracy" in r]
                    if valid_scenario1:
                        avg_acc1 = sum([r.get("accuracy", 0) for r in valid_scenario1]) / len(valid_scenario1)
                        print(f"\n📊 当前进度：{idx+1}/{len(self.test_cases)}")
                        print(f"  场景1平均准确率：{avg_acc1:.4f}")
                
                if scenario2_results:
                    valid_scenario2 = [r for r in scenario2_results if "accuracy" in r]
                    if valid_scenario2:
                        avg_acc2 = sum([r["accuracy"] for r in valid_scenario2]) / len(valid_scenario2)
                        avg_recall2 = sum([r.get("recall_score", 0) for r in valid_scenario2]) / len(valid_scenario2)
                        print(f"  场景2平均准确率：{avg_acc2:.4f}，平均召回率：{avg_recall2:.4f}")
            
            # 添加延迟，避免GPU过载
            time.sleep(0.2)
        
        # 保存测试结果
        llm_results = {
            "model_config": self.model_config,
            "model_name": self.model_name,
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases_count": len(self.test_cases),
            "scenario1_results": scenario1_results,
            "scenario2_results": scenario2_results,
            "summary": self._calculate_summary(scenario1_results, scenario2_results)
        }
        
        result_file = os.path.join(self.output_dir, "results", f"{self.model_name}_test_results.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(llm_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ {self.model_name} 模型测试完成")
        print(f"   场景1平均准确率: {llm_results['summary']['scenario1_avg_accuracy']:.4f}")
        print(f"   场景2平均准确率: {llm_results['summary']['scenario2_avg_accuracy']:.4f}")
        print(f"   场景2平均召回率: {llm_results['summary']['scenario2_avg_recall']:.4f}")
        print(f"   结果文件: {result_file}")
        
        # 生成简要统计报告
        self._generate_brief_statistics(llm_results)
        
        # 清理资源
        ModelManager.cleanup_models(llm_model, embedding_models)
        if index is not None:
            del index
        if 'enhanced_docs' in locals():
            del enhanced_docs
        
        print(f"\n{'='*60}")
        print(f"{self.model_name} 模型测试完成")
        print(f"结果目录: {self.output_dir}")
        print(f"{'='*60}")
    
    def _calculate_summary(self, scenario1_results, scenario2_results):
        """计算测试结果摘要"""
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
        """生成简要统计报告"""
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
            scenario1_fail = len(llm_results['scenario1_results']) - scenario1_success
            
            scenario2_success = sum(1 for r in llm_results['scenario2_results'] if r.get("accuracy", 0) == 1)
            scenario2_fail = len(llm_results['scenario2_results']) - scenario2_success
            
            f.write("📈 成功/失败统计:\n")
            f.write(f"   场景1: 成功 {scenario1_success} 条，失败 {scenario1_fail} 条\n")
            f.write(f"   场景2: 成功 {scenario2_success} 条，失败 {scenario2_fail} 条\n\n")
            
            # 错误分析
            f.write("⚠️  错误分析:\n")
            errors = []
            for result in llm_results['scenario1_results'] + llm_results['scenario2_results']:
                if "error" in result:
                    errors.append(result["error"])
            
            if errors:
                unique_errors = list(set(errors))
                f.write(f"   共发现 {len(errors)} 个错误，{len(unique_errors)} 种类型:\n")
                for error in unique_errors[:5]:  # 只显示前5种错误类型
                    f.write(f"     - {error[:100]}\n")
            else:
                f.write("   无错误记录\n")
        
        print(f"📊 简要统计已保存: {stats_file}")

# ====================== 主函数 ======================
def main():
    """主函数：运行模型测试"""
    
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