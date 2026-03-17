import torch
import json
import numpy as np
import faiss
import os
import re
import gc
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import sys

# 日志级别控制
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE_PATH = None

def log(message: str, level: str = "INFO") -> None:
    """日志记录函数"""
    levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    if levels[level] >= levels[LOG_LEVEL]:
        log_str = f"[{level}] {message}"
        print(log_str)
        # 写入日志文件
        if LOG_FILE_PATH:
            try:
                with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                    f.write(log_str + '\n')
            except Exception as e:
                print(f"[ERROR] 写入日志文件失败: {e}")

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类"""
    def __init__(self, llm_name, llm_local_path, embedding_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path
        self.embedding_local_path = embedding_local_path

# ====================== Qwen模型配置 ======================
class QwenTestConfig:
    """Qwen模型测试配置"""
    def __init__(self):
        # Qwen模型配置
        self.llm_config = {
            "llm_name": "Qwen2.5-3B-Instruct",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct",
            "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5/BAAI/bge-large-zh-v1___5"
        }
        
        # 测试数据路径
        self.test_data_path = "qa_data/520_qa.json"
        self.knowledge_base_path = "qa_data/knowledge_base.txt"
        
        # 输出目录（包含模型名称）
        self.output_dir = f"./qwen_3b_model_520qa"
        
        # 测试参数
        self.max_test_cases = 520
        self.batch_size = 8  # 增加批量大小，提高GPU利用率
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.75
        
        # 推理参数
        self.max_new_tokens = 150  # 适当减少token数，提高生成速度
        self.rag_max_new_tokens = 200  # 适当减少token数，提高生成速度
        self.temperature = 0.1
        self.top_p = 0.9
        self.do_sample = False  # 使用贪婪解码，提高生成速度
        self.repetition_penalty = 1.1
        self.no_repeat_ngram_size = 3
        
        # 性能优化参数
        self.enable_batching = True
        self.log_interval = 20  # 减少日志频率，每20个批次输出一次
        self.memory_efficient_loading = True

# ====================== 模型管理器 ======================
class ModelManager:
    """模型管理器"""
    
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
        加载本地模型
        返回: (tokenizer, llm_model, embedding_models)
        """
        log(f"正在加载Qwen模型：{config['llm_name']}", "INFO")
        
        try:
            # 1. 加载大模型tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                padding_side="left",
                local_files_only=True,
                use_fast=True  # 使用快速tokenizer
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 2. 加载大模型 - 启用Flash Attention（如果支持）
            try:
                llm_model = AutoModelForCausalLM.from_pretrained(
                    config["llm_local_path"],
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=ModelManager.get_bnb_config(),
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    offload_folder="./offload",  # 内存不足时的卸载目录
                    attn_implementation="flash_attention_2"  # 启用Flash Attention加速
                )
                log("✅ Flash Attention已启用，将加速模型推理", "INFO")
            except Exception as e:
                # 如果Flash Attention不支持，回退到默认实现
                log(f"⚠️  Flash Attention不可用，使用默认实现: {e}", "WARNING")
                llm_model = AutoModelForCausalLM.from_pretrained(
                    config["llm_local_path"],
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=ModelManager.get_bnb_config(),
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    offload_folder="./offload"  # 内存不足时的卸载目录
                )
            llm_model.eval()
            log(f"✅ Qwen模型 {config['llm_name']} 加载成功", "INFO")
        except Exception as e:
            log(f"❌ Qwen模型加载失败，错误信息：{e}", "ERROR")
            return None, None, None
        
        # 3. 加载Embedding模型
        log(f"正在加载BGE Embedding模型", "INFO")
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True  # 使用快速tokenizer
            )
            embedding_model = AutoModel.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            embedding_model.eval()
            log(f"✅ BGE Embedding模型加载成功", "INFO")
        except Exception as e:
            log(f"❌ BGE Embedding模型加载失败，错误信息：{e}", "ERROR")
            return None, None, None
        
        return tokenizer, llm_model, (embedding_tokenizer, embedding_model)
    
    @staticmethod
    def cleanup_models(llm_model, embedding_models=None):
        """清理模型资源"""
        try:
            if llm_model is not None:
                del llm_model
            
            if embedding_models is not None:
                embedding_tokenizer, embedding_model = embedding_models
                del embedding_tokenizer
                del embedding_model
            
            torch.cuda.empty_cache()
            gc.collect()
            log("✅ 模型资源已清理", "INFO")
        except Exception as e:
            log(f"⚠️  模型资源清理时发生错误：{e}", "WARNING")

# ====================== BGE文本转向量函数 ======================
def bge_embedding_encode(embedding_models, text: Union[str, List[str]], batch_mode: bool = False) -> np.ndarray:
    """BGE文本转向量函数
    
    Args:
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        text: 单个文本字符串或文本列表
        batch_mode: 是否使用批量模式
    
    Returns:
        np.ndarray: 文本的向量表示
    """
    embedding_tokenizer, embedding_model = embedding_models
    
    # 处理空输入
    if not text:
        if batch_mode:
            return np.empty((0, 1024))  # 返回空数组，维度与模型输出一致
        return np.array([])
    
    # 确保文本是列表格式（用于统一处理）
    if not batch_mode and not isinstance(text, list):
        text = [str(text).strip()]
    elif isinstance(text, list):
        text = [str(t).strip() for t in text]
    
    # 过滤空字符串
    text = [t for t in text if t]
    if not text:
        if batch_mode:
            return np.empty((0, 1024))
        return np.array([])
    
    max_length = 512
    
    # 批量处理
    inputs = embedding_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    ).to(embedding_model.device)
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        
        # 适配BGE-M3模型，检查是否有pooler_output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0]
        
        # 归一化
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # 转换为numpy数组
    embeddings_np = embeddings.cpu().numpy()
    
    # 如果不是批量模式且只有一个输入，返回一维数组
    if not batch_mode and embeddings_np.shape[0] == 1:
        return embeddings_np.squeeze()
    
    return embeddings_np

# ====================== 构建FAISS向量索引 ======================
def build_vector_index(embedding_models, docs: List[str], batch_size: int = 32) -> Tuple[Optional[faiss.Index], Optional[List[str]]]:
    """构建FAISS向量索引
    
    Args:
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        docs: 文档列表
        batch_size: 批处理大小
    
    Returns:
        Tuple[Optional[faiss.Index], Optional[List[str]]]: FAISS索引和有效文档列表
    """
    doc_vectors = []
    valid_docs = []
    
    total_docs = len(docs)
    log(f"正在编码 {total_docs} 条文档...", "INFO")
    
    # 过滤空文档
    docs = [doc for doc in docs if doc.strip()]
    if not docs:
        log("❌ 无有效文档，无法构建FAISS索引", "ERROR")
        return None, None
    
    start_time = time.time()
    
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_vectors = bge_embedding_encode(embedding_models, batch_docs, batch_mode=True)
        
        # 批量添加，避免循环
        if batch_vectors.size > 0:
            doc_vectors.append(batch_vectors)
            valid_docs.extend(batch_docs)
        
        # 日志输出，减少打印频率
        if (i // batch_size + 1) % 20 == 0:
            elapsed_time = time.time() - start_time
            processed = min(i+batch_size, len(docs))
            log(f"  已编码 {processed}/{len(docs)} 条文档，耗时: {elapsed_time:.2f}s", "INFO")
    
    if not doc_vectors:
        log("❌ 无有效文档向量，无法构建FAISS索引", "ERROR")
        return None, None
    
    # 合并向量
    doc_vectors = np.concatenate(doc_vectors, axis=0).astype(np.float32)
    vector_dim = doc_vectors.shape[1]
    
    # 构建FAISS索引
    index = faiss.IndexFlatIP(vector_dim)
    index.add(doc_vectors)
    
    # 构建倒排索引，用于加速关键词检索
    build_inverted_index(valid_docs)
    
    elapsed_time = time.time() - start_time
    log(f"✅ 共构建 {len(valid_docs)} 条有效文档的FAISS索引，耗时: {elapsed_time:.2f}s", "INFO")
    log(f"  向量维度: {vector_dim}", "INFO")
    log(f"  索引类型: IndexFlatIP (内积相似度)", "INFO")
    log(f"  倒排索引已构建，用于加速关键词检索", "INFO")
    
    return index, valid_docs

# ====================== 检索函数 ======================
# 问题向量缓存，减少重复计算
question_vector_cache = {}

def enhanced_retrieval(embedding_models, index, docs: List[str], question: str, top_k: int = 5, similarity_threshold: float = 0.75) -> List[str]:
    """改进的检索函数
    Args:
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        index: FAISS索引
        docs: 文档列表
        question: 查询问题
        top_k: 返回的文档数量
        similarity_threshold: 相似度阈值

    Returns:
        List[str]: 检索到的文档列表
    """
    results = []
    
    # 1. 向量检索 - 使用缓存优化
    global question_vector_cache
    question_key = question.strip()
    
    # 检查缓存中是否已有该问题的向量
    if question_key in question_vector_cache:
        question_vector = question_vector_cache[question_key]
    else:
        # 计算新的问题向量并缓存
        question_vector = bge_embedding_encode(embedding_models, question)
        if question_vector.size > 0:
            question_vector_cache[question_key] = question_vector
    
    if question_vector.size == 0:
        return []
    
    # 搜索更多候选文档，然后过滤
    distances, indices = index.search(question_vector.reshape(1, -1), top_k * 10)
    
    # 批量处理结果
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(docs) and dist > similarity_threshold:
            results.append({
                "doc": docs[idx],
                "similarity": float(dist),
                "type": "vector",
                "index": idx
            })
    
    # 2. 关键词检索（后备）
    if len(results) < 2:
        keyword_results = keyword_based_retrieval(question, docs)
        results.extend(keyword_results)
    
    # 3. 去重和排序
    seen_indices = set()
    unique_results = []
    
    for result in results:
        idx = result.get("index", -1)
        if idx >= 0 and idx not in seen_indices:
            seen_indices.add(idx)
            unique_results.append(result)
    
    # 排序并返回前top_k个结果
    unique_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return [r["doc"] for r in unique_results[:top_k]]

# 倒排索引缓存，提高关键词检索效率
inverted_index = None

# 初始化倒排索引
def build_inverted_index(docs: List[str]):
    """构建倒排索引，用于加速关键词检索
    
    Args:
        docs: 文档列表
    """
    global inverted_index
    inverted_index = {}
    
    for idx, doc in enumerate(docs):
        # 提取文档中的关键词
        doc_keywords = set()
        
        # 提取公司名
        companies = extract_company_names(doc)
        doc_keywords.update(companies)
        
        # 提取其他关键词
        keywords = extract_keywords(doc)
        doc_keywords.update(keywords)
        
        # 构建倒排索引
        for keyword in doc_keywords:
            if keyword not in inverted_index:
                inverted_index[keyword] = []
            inverted_index[keyword].append(idx)

def keyword_based_retrieval(query: str, docs: List[str]) -> List[Dict[str, Any]]:
    """关键词检索（后备）
    
    Args:
        query: 查询问题
        docs: 文档列表
    
    Returns:
        List[Dict]: 检索结果列表
    """
    results = []
    
    # 提取关键词
    companies = extract_company_names(query)
    keywords = extract_keywords(query)
    
    # 提前计算是否有公司或关键词
    has_companies = len(companies) > 0
    has_keywords = len(keywords) > 0
    
    if not has_companies and not has_keywords:
        # 清理GPU缓存
        torch.cuda.empty_cache()
        return results
    
    # 使用倒排索引加速关键词匹配
    global inverted_index
    if inverted_index is None:
        build_inverted_index(docs)
    
    # 收集相关文档索引
    relevant_indices = set()
    if has_companies:
        for company in companies:
            if company in inverted_index:
                relevant_indices.update(inverted_index[company])
    
    if has_keywords:
        for keyword in keywords:
            if keyword in inverted_index:
                relevant_indices.update(inverted_index[keyword])
    
    # 如果没有找到相关文档，返回空结果
    if not relevant_indices:
        return results
    
    # 计算相关文档的分数
    for idx in relevant_indices:
        if idx >= len(docs):
            continue
        
        doc = docs[idx]
        score = 0
        
        if has_companies:
            for company in companies:
                if company in doc:
                    score += 1.0
                    break
        
        if has_keywords:
            for keyword in keywords:
                if keyword in doc:
                    score += 0.5
        
        if score > 0:
            results.append({
                "doc": doc,
                "similarity": min(score / (2.0 if has_companies and has_keywords else 1.0), 1.0),
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
        "地址", "注册地址", "经营地址"
    ]
    
    for keyword in bid_keywords:
        if keyword in text:
            keywords.append(keyword)
    
    return keywords

# ====================== 推理函数 ======================
def direct_inference_no_prompt(tokenizer, llm_model, question, max_new_tokens=200, temperature=0.1, top_p=0.9, do_sample=False):
    """场景1：无任何提示词，直接让模型回答问题
    
    Args:
        tokenizer: 模型tokenizer
        llm_model: 大语言模型
        question: 查询问题
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: 核采样参数
        do_sample: 是否使用采样解码
    
    Returns:
        Tuple[str, List]: 模型回答和检索文档列表（此处为空）
    """
    inputs = tokenizer(
        question,  # 只输入问题，不加任何提示
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=True  # 使用缓存提高速度
        )
    
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    return model_answer, []  # 无检索文档

# 批量推理支持
def batch_inference_no_prompt(tokenizer, llm_model, questions, max_new_tokens=200, temperature=0.1, top_p=0.9, do_sample=False):
    """批量进行无提示词推理
    
    Args:
        tokenizer: 模型tokenizer
        llm_model: 大语言模型
        questions: 查询问题列表
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: 核采样参数
        do_sample: 是否使用采样解码
    
    Returns:
        List[Tuple[str, List]]: 模型回答和检索文档列表（此处为空）的列表
    """
    if not questions:
        return []
    
    # 批量编码
    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=True
        )
    
    # 批量解码
    model_answers = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    return [(answer, []) for answer in model_answers]

def optimized_rag_inference(tokenizer, llm_model, embedding_models, index, docs, question, top_k=5, similarity_threshold=0.75, max_new_tokens=300, temperature=0.1, top_p=0.9, do_sample=False):
    """场景2：使用专业提示词模板的RAG推理
    
    Args:
        tokenizer: 模型tokenizer
        llm_model: 大语言模型
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        index: FAISS索引
        docs: 文档列表
        question: 查询问题
        top_k: 检索文档数量
        similarity_threshold: 相似度阈值
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: 核采样参数
        do_sample: 是否使用采样解码
    
    Returns:
        Tuple[str, List[str]]: 模型回答和检索文档列表
    """
    retrieved_docs = enhanced_retrieval(embedding_models, index, docs, question, top_k=top_k, similarity_threshold=similarity_threshold)
    
    if not retrieved_docs:
        return "根据现有信息无法确定。", []
    
    context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
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

# 现在请根据以下信息回答问题
{context}

# 用户问题
{question}

请根据以上信息和示例风格，按照回答要求给出专业回答。
答："""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # 可能需要增加长度限制
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=True  # 使用缓存提高速度
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # 从响应中提取模型回答部分
    answer_marker = "答："
    if answer_marker in full_response:
        # 找到最后一个"答："的位置
        last_answer_pos = full_response.rfind(answer_marker)
        model_answer = full_response[last_answer_pos + len(answer_marker):].strip()
    else:
        # 如果找不到"答："标记，则返回整个响应
        model_answer = full_response.strip()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    return model_answer, retrieved_docs

def batch_rag_inference(tokenizer, llm_model, embedding_models, index, docs, questions, top_k=5, similarity_threshold=0.75, max_new_tokens=300, temperature=0.1, top_p=0.9, do_sample=False):
    """批量进行RAG推理
    
    Args:
        tokenizer: 模型tokenizer
        llm_model: 大语言模型
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        index: FAISS索引
        docs: 文档列表
        questions: 查询问题列表
        top_k: 检索文档数量
        similarity_threshold: 相似度阈值
        max_new_tokens: 最大生成token数
        temperature: 生成温度
        top_p: 核采样参数
        do_sample: 是否使用采样解码
    
    Returns:
        List[Tuple[str, List[str]]]: 模型回答和检索文档列表的列表
    """
    if not questions:
        return []
    
    # 1. 批量检索所有问题的相关文档
    all_retrieved_docs = []
    for question in questions:
        retrieved_docs = enhanced_retrieval(embedding_models, index, docs, question, 
                                           top_k=top_k, similarity_threshold=similarity_threshold)
        all_retrieved_docs.append(retrieved_docs)
    
    # 2. 构建所有问题的prompt
    prompts = []
    for question, retrieved_docs in zip(questions, all_retrieved_docs):
        if not retrieved_docs:
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

# 现在请根据你的知识回答问题

# 用户问题
{question}

请按照回答要求给出专业回答。
答："""
        else:
            context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
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

# 现在请根据以下信息回答问题
{context}

# 用户问题
{question}

请根据以上信息和示例风格，按照回答要求给出专业回答。
答："""
        
        prompts.append(prompt)
    
    # 3. 批量生成回答
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=True
        )
    
    # 4. 批量解码并处理结果
    results = []
    for i, output in enumerate(outputs):
        full_response = tokenizer.decode(output, skip_special_tokens=True).strip()
        retrieved_docs = all_retrieved_docs[i]
        
        # 从响应中提取模型回答部分
        answer_marker = "答："
        if answer_marker in full_response:
            # 找到最后一个"答："的位置
            last_answer_pos = full_response.rfind(answer_marker)
            model_answer = full_response[last_answer_pos + len(answer_marker):].strip()
        else:
            # 如果找不到"答："标记，则返回整个响应
            model_answer = full_response.strip()
        
        results.append((model_answer, retrieved_docs))
    
    return results

# ====================== 数据加载器 ======================
def load_qa_data(qa_file_path="qa_data/100_qa.json", kb_file_path="qa_data/knowledge_base.txt"):
    """加载QA数据
    
    Args:
        qa_file_path: QA数据文件路径
        kb_file_path: 知识库文件路径
    
    Returns:
        Tuple[List[Dict], List[str]]: 测试用例列表和知识库文档列表
    """
    try:
        # 1. 加载知识库文档
        knowledge_docs = []
        if os.path.exists(kb_file_path):
            with open(kb_file_path, "r", encoding="utf-8") as f:
                knowledge_docs = [line.strip() for line in f if line.strip()]
            log(f"✅ 加载 {len(knowledge_docs)} 条知识库文档", "INFO")
        else:
            log("⚠️  未找到知识库文件，将从问答对中构建", "WARNING")
        
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
                relevant_docs = [doc.strip() for doc in item["relevant_documents"] if doc.strip()]
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
        
        log(f"✅ 加载 {len(test_cases)} 条有效测试用例", "INFO")
        
        # 3. 合并知识库（如果知识库为空）
        if not knowledge_docs:
            all_docs_set = set()
            for case in test_cases:
                all_docs_set.update(case["relevant_docs"])
            knowledge_docs = list(all_docs_set)
            log(f"📚 从问答对构建 {len(knowledge_docs)} 条知识库文档", "INFO")
        
        return test_cases, knowledge_docs
    
    except FileNotFoundError:
        log(f"❌ 未找到文件：{qa_file_path}", "ERROR")
        return [], []
    except json.JSONDecodeError as e:
        log(f"❌ JSON解析失败：{e}", "ERROR")
        return [], []
    except Exception as e:
        log(f"❌ 加载问答对失败：{e}", "ERROR")
        return [], []

# ====================== 评估函数 ======================
# 预编译正则表达式，提高性能
COMPANY_PATTERNS = [
    re.compile(r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)'),
    re.compile(r'供应商[：:]?([\u4e00-\u9fa5a-zA-Z0-9]{2,})'),
    re.compile(r'由([\u4e00-\u9fa5a-zA-Z0-9]{2,})提供')
]

PRICE_PATTERN = re.compile(r'(\d+\.?\d*)元')

PRODUCT_INDICATORS = {"产品", "设备", "仪器", "系统", "项目", "服务"}

IMPORTANT_KEYWORDS = {"法定代表人", "公司", "地址", "金额", "供应商", "采购方", "中标", "价格", "项目"}

ERROR_PREFIXES = {"对不起", "抱歉", "我不确定", "无法回答", "我不知道"}

def extract_entities_from_text(text):
    """从文本中提取实体
    
    Args:
        text: 输入文本
    
    Returns:
        List[str]: 实体列表
    """
    entities = set()  # 使用set避免重复
    
    # 提取公司名
    for pattern in COMPANY_PATTERNS:
        matches = pattern.findall(text)
        entities.update(matches)
    
    # 提取产品相关实体
    for word in text.split():
        if any(indicator in word for indicator in PRODUCT_INDICATORS):
            entities.add(word)
    
    # 提取价格
    price_matches = PRICE_PATTERN.findall(text)
    entities.update(price_matches)
    
    return list(entities)

def is_doc_related(doc1, doc2):
    """检查文档是否相关
    
    Args:
        doc1: 文档1
        doc2: 文档2
    
    Returns:
        bool: 是否相关
    """
    # 快速检查：如果有完全匹配的情况
    if doc1 == doc2:
        return True
    
    # 检查是否有共同实体（快速方式）
    entities1 = extract_entities_from_text(doc1)
    entities2 = extract_entities_from_text(doc2)
    
    # 检查是否有共同实体
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    # 简化的相似度检查：使用关键词匹配代替SequenceMatcher
    # 提取关键词并检查重叠
    doc1_lower = doc1.lower()
    doc2_lower = doc2.lower()
    
    # 检查重要关键词是否重叠
    for keyword in IMPORTANT_KEYWORDS:
        if keyword.lower() in doc1_lower and keyword.lower() in doc2_lower:
            return True
    
    return False

def calculate_recall(retrieved_docs, relevant_docs):
    """计算召回率
    
    Args:
        retrieved_docs: 检索到的文档列表
        relevant_docs: 相关文档列表
    
    Returns:
        float: 召回率
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # 快速检查：如果有任何相关文档被检索到
    for retrieved_doc in retrieved_docs:
        for relevant_doc in relevant_docs:
            if is_doc_related(retrieved_doc, relevant_doc):
                return 1.0
    
    return 0.0

def calculate_accuracy(model_answer, reference_answer, threshold=0.6):
    """计算准确率
    
    Args:
        model_answer: 模型回答
        reference_answer: 参考回答
        threshold: 相似度阈值
    
    Returns:
        float: 准确率
    """
    if not model_answer or not reference_answer:
        return 0.0
    
    # 简单匹配检查（快速）
    if reference_answer in model_answer or model_answer in reference_answer:
        return 1.0
    
    # 关键词匹配（快速）
    match_count = 0
    total_keywords = 0
    
    # 只检查参考回答中存在的关键词
    reference_lower = reference_answer.lower()
    model_lower = model_answer.lower()
    
    for keyword in IMPORTANT_KEYWORDS:
        keyword_lower = keyword.lower()
        if keyword_lower in reference_lower:
            total_keywords += 1
            if keyword_lower in model_lower:
                match_count += 1
    
    if total_keywords > 0:
        keyword_ratio = match_count / total_keywords
        if keyword_ratio >= threshold:
            return 1.0
    
    # 实体匹配（快速）
    ref_entities = extract_entities_from_text(reference_answer)
    if ref_entities:
        model_entities = extract_entities_from_text(model_answer)
        common_entities = set(ref_entities) & set(model_entities)
        if len(common_entities) / len(ref_entities) >= 0.5:
            return 1.0
    
    return 0.0

def calculate_answer_quality(model_answer, reference_answer):
    """
    评估回答质量：包括相关性、完整性、一致性
    返回一个综合质量分数（0-1）
    
    Args:
        model_answer: 模型回答
        reference_answer: 参考回答
    
    Returns:
        Dict: 质量评估结果
    """
    # 1. 简化的文本相似度检查
    # 使用更快速的字符串匹配方法，避免SequenceMatcher的O(n²)复杂度
    reference_lower = reference_answer.lower()
    model_lower = model_answer.lower()
    
    # 计算共同词的比例
    ref_words = set(reference_lower.split())
    model_words = set(model_lower.split())
    
    if ref_words:
        word_overlap = len(ref_words & model_words) / len(ref_words)
    else:
        word_overlap = 0.0
    
    # 2. 检查是否包含关键信息（快速）
    keyword_hit = 0
    total_ref_keywords = 0
    
    for keyword in IMPORTANT_KEYWORDS:
        keyword_lower = keyword.lower()
        if keyword_lower in reference_lower:
            total_ref_keywords += 1
            if keyword_lower in model_lower:
                keyword_hit += 1
    
    keyword_score = keyword_hit / total_ref_keywords if total_ref_keywords > 0 else 0
    
    # 3. 检查回答格式（快速）
    format_score = 1.0
    for prefix in ERROR_PREFIXES:
        if model_answer.startswith(prefix):
            format_score -= 0.2
            break  # 只扣一次分
    
    # 4. 计算综合分数
    final_score = (word_overlap * 0.4) + (keyword_score * 0.4) + (format_score * 0.2)
    
    return {
        "quality_score": final_score,
        "similarity": word_overlap,  # 使用词重叠率代替SequenceMatcher
        "keyword_score": keyword_score,
        "format_score": format_score
    }

# ====================== 测试运行器 ======================
class QwenModelTestRunner:
    """Qwen模型测试运行器"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = self._setup_output_dir()
        self.test_cases, self.knowledge_docs = load_qa_data(
            config.test_data_path, 
            config.knowledge_base_path
        )
        
        # 限制测试用例数量
        if len(self.test_cases) > config.max_test_cases:
            self.test_cases = self.test_cases[:config.max_test_cases]
            log(f"📊 限制测试用例数为: {len(self.test_cases)}", "INFO")
        
        log(f"\n{'='*60}", "INFO")
        log(f"Qwen模型测试初始化完成", "INFO")
        log(f"模型名称: {config.llm_config['llm_name']}", "INFO")
        log(f"测试用例数: {len(self.test_cases)}", "INFO")
        log(f"知识库文档数: {len(self.knowledge_docs)}", "INFO")
        log(f"输出目录: {self.output_dir}", "INFO")
        log(f"{'='*60}\n", "INFO")
    
    def _setup_output_dir(self):
        """设置输出目录"""
        # 使用模型名称和时间戳创建唯一目录
        model_name = self.config.llm_config['llm_name'].replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.output_dir, f"{model_name}_{timestamp}")
        
        # 只创建llm_results和logs目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "llm_results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # 初始化日志文件路径
        global LOG_FILE_PATH
        LOG_FILE_PATH = os.path.join(output_dir, "logs", f"test_{timestamp}.log")
        
        return output_dir
    
    def run_qwen_tests(self):
        """运行Qwen模型测试"""
        log(f"\n{'='*60}", "INFO")
        log(f"开始Qwen模型测试", "INFO")
        log(f"模型: {self.config.llm_config['llm_name']}", "INFO")
        log(f"{'='*60}", "INFO")
        
        start_time = time.time()
        
        # 加载Qwen模型
        tokenizer, llm_model, embedding_models = ModelManager.load_local_models(self.config.llm_config)
        if llm_model is None:
            log(f"❌ Qwen模型加载失败，测试终止", "ERROR")
            return
        
        # 构建向量索引
        log("\n正在构建FAISS向量索引...", "INFO")
        index, enhanced_docs = build_vector_index(embedding_models, self.knowledge_docs, 
                                                  batch_size=self.config.batch_size)
        if index is None:
            log("⚠️  向量索引构建失败，场景2将无法测试", "WARNING")
        
        # 运行测试
        scenario1_results = []
        scenario2_results = []
        
        # 测试用例批量处理支持
        batch_size = self.config.batch_size
        total_test_cases = len(self.test_cases)
        
        for idx in range(0, total_test_cases, batch_size):
            batch_test_cases = self.test_cases[idx:idx+batch_size]
            
            # 场景1：无提示词直接回答 - 支持批量处理
            try:
                log(f"  正在处理场景1（无提示词）：批量 {idx//batch_size + 1}/{total_test_cases//batch_size + 1}", "INFO")
                questions = [case["question"] for case in batch_test_cases]
                
                log(f"    开始批量推理，问题数量：{len(questions)}", "INFO")
                batch_answers = batch_inference_no_prompt(
                    tokenizer, llm_model, questions,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample
                )
                log(f"    批量推理完成，获得 {len(batch_answers)} 个回答", "INFO")
                
                # 处理批量结果
                for i, (model_answer1, _) in enumerate(batch_answers):
                    case_idx = idx + i
                    test_case = batch_test_cases[i]
                    reference_answer = test_case["reference_answer"]
                    
                    accuracy1 = calculate_accuracy(model_answer1, reference_answer)
                    quality_metrics1 = calculate_answer_quality(model_answer1, reference_answer)
                    
                    result1 = {
                        "scenario": "no_prompt",
                        "test_case_id": case_idx + 1,
                        "question": test_case["question"],
                        "reference_answer": reference_answer,
                        "model_answer": model_answer1,
                        "accuracy": accuracy1,
                        "answer_length": len(model_answer1),
                        "quality_score": quality_metrics1["quality_score"],
                        "similarity": quality_metrics1["similarity"],
                        "keyword_score": quality_metrics1["keyword_score"]
                    }
                    scenario1_results.append(result1)
                
            except Exception as e:
                log(f"❌ 场景1批量测试失败：{e}", "ERROR")
                # 回退到单样本处理
                for i, test_case in enumerate(batch_test_cases):
                    case_idx = idx + i
                    try:
                        model_answer1, _ = direct_inference_no_prompt(
                            tokenizer, llm_model, test_case["question"],
                            max_new_tokens=self.config.max_new_tokens,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            do_sample=self.config.do_sample
                        )
                        accuracy1 = calculate_accuracy(model_answer1, test_case["reference_answer"])
                        quality_metrics1 = calculate_answer_quality(model_answer1, test_case["reference_answer"])
                        
                        result1 = {
                            "scenario": "no_prompt",
                            "test_case_id": case_idx + 1,
                            "question": test_case["question"],
                            "reference_answer": test_case["reference_answer"],
                            "model_answer": model_answer1,
                            "accuracy": accuracy1,
                            "answer_length": len(model_answer1),
                            "quality_score": quality_metrics1["quality_score"],
                            "similarity": quality_metrics1["similarity"],
                            "keyword_score": quality_metrics1["keyword_score"]
                        }
                        scenario1_results.append(result1)
                    except Exception as e2:
                        scenario1_results.append({
                            "scenario": "no_prompt",
                            "test_case_id": case_idx + 1,
                            "question": test_case["question"],
                            "error": str(e2)
                        })
            
            # 场景2：有提示词的RAG回答 - 使用批量处理
            if index is not None:
                log(f"  正在处理场景2（RAG）：批量 {idx//batch_size + 1}/{total_test_cases//batch_size + 1}", "INFO")
                try:
                    questions = [case["question"] for case in batch_test_cases]
                    
                    log(f"    开始批量RAG推理，问题数量：{len(questions)}", "INFO")
                    batch_answers = batch_rag_inference(
                        tokenizer, llm_model, embedding_models, index, enhanced_docs, questions,
                        top_k=self.config.top_k_retrieval,
                        similarity_threshold=self.config.similarity_threshold,
                        max_new_tokens=self.config.rag_max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample
                    )
                    log(f"    批量RAG推理完成，获得 {len(batch_answers)} 个回答", "INFO")
                    
                    # 处理批量结果
                    for i, (model_answer2, retrieved_docs) in enumerate(batch_answers):
                        case_idx = idx + i
                        test_case = batch_test_cases[i]
                        reference_answer = test_case["reference_answer"]
                        relevant_docs = test_case["relevant_docs"]
                        
                        recall2 = calculate_recall(retrieved_docs, relevant_docs)
                        accuracy2 = calculate_accuracy(model_answer2, reference_answer)
                        quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
                        
                        result2 = {
                            "scenario": "with_prompt_rag",
                            "test_case_id": case_idx + 1,
                            "question": questions[i],
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
                        scenario2_results.append(result2)
                        
                except Exception as e:
                    log(f"❌ 场景2批量测试失败：{e}", "ERROR")
                    # 回退到单样本处理
                    for i, test_case in enumerate(batch_test_cases):
                        case_idx = idx + i
                        question = test_case["question"]
                        reference_answer = test_case["reference_answer"]
                        relevant_docs = test_case["relevant_docs"]
                        
                        try:
                            model_answer2, retrieved_docs = optimized_rag_inference(
                                tokenizer, llm_model, embedding_models, index, enhanced_docs, question,
                                top_k=self.config.top_k_retrieval,
                                similarity_threshold=self.config.similarity_threshold,
                                max_new_tokens=self.config.rag_max_new_tokens,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                do_sample=self.config.do_sample
                            )
                            
                            recall2 = calculate_recall(retrieved_docs, relevant_docs)
                            accuracy2 = calculate_accuracy(model_answer2, reference_answer)
                            quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
                            
                            result2 = {
                                "scenario": "with_prompt_rag",
                                "test_case_id": case_idx + 1,
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
                            scenario2_results.append(result2)
                            
                        except Exception as e2:
                            scenario2_results.append({
                                "scenario": "with_prompt_rag",
                                "test_case_id": case_idx + 1,
                                "question": question,
                                "error": str(e2)
                            })
            else:
                for i, test_case in enumerate(batch_test_cases):
                    case_idx = idx + i
                    scenario2_results.append({
                        "scenario": "with_prompt_rag",
                        "test_case_id": case_idx + 1,
                        "question": test_case["question"],
                        "error": "FAISS索引未构建"
                    })
            
            # 清理GPU缓存，优化内存利用
            torch.cuda.empty_cache()
            gc.collect()
            
            # 进度报告
            processed = min(idx + batch_size, total_test_cases)
            if processed % self.config.log_interval == 0 or processed == total_test_cases:
                elapsed = time.time() - start_time
                log(f"\n📊 当前进度：{processed}/{total_test_cases}，耗时: {elapsed:.2f}s", "INFO")
                
                if scenario1_results:
                    valid_results1 = [r for r in scenario1_results if "accuracy" in r]
                    if valid_results1:
                        avg_acc1 = sum([r["accuracy"] for r in valid_results1]) / len(valid_results1)
                        log(f"  场景1平均准确率：{avg_acc1:.4f}", "INFO")
                
                if scenario2_results:
                    valid_results2 = [r for r in scenario2_results if "accuracy" in r]
                    if valid_results2:
                        avg_acc2 = sum([r["accuracy"] for r in valid_results2]) / len(valid_results2)
                        avg_recall2 = sum([r.get("recall_score", 0) for r in valid_results2]) / len(valid_results2)
                        log(f"  场景2平均准确率：{avg_acc2:.4f}，平均召回率：{avg_recall2:.4f}", "INFO")
        
        # 保存Qwen模型的测试结果
        qwen_results = {
            "llm_config": self.config.llm_config,
            "test_config": {
                "test_cases_count": len(self.test_cases),
                "knowledge_docs_count": len(self.knowledge_docs),
                "max_test_cases": self.config.max_test_cases,
                "batch_size": self.config.batch_size,
                "top_k_retrieval": self.config.top_k_retrieval,
                "similarity_threshold": self.config.similarity_threshold
            },
            "scenario1_results": scenario1_results,
            "scenario2_results": scenario2_results,
            "summary": self._calculate_summary(scenario1_results, scenario2_results),
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": time.time() - start_time
        }
        
        # 保存结果文件
        result_file = os.path.join(self.output_dir, "llm_results", 
                                  f"qwen_test_results_{datetime.now().strftime('%H%M%S')}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(qwen_results, f, ensure_ascii=False, indent=2)
        
        total_time = time.time() - start_time
        log(f"\n✅ Qwen模型测试完成，总耗时: {total_time:.2f}s", "INFO")
        log(f"   场景1平均准确率: {qwen_results['summary']['scenario1_avg_accuracy']:.4f}", "INFO")
        log(f"   场景2平均准确率: {qwen_results['summary']['scenario2_avg_accuracy']:.4f}", "INFO")
        log(f"   场景2平均召回率: {qwen_results['summary']['scenario2_avg_recall']:.4f}", "INFO")
        log(f"   结果文件: {result_file}", "INFO")
        
        # 生成日志文件
        self._generate_log_file(qwen_results)
        
        # 清理资源
        ModelManager.cleanup_models(llm_model, embedding_models)
        
        # 清理缓存
        global question_vector_cache, inverted_index
        question_vector_cache.clear()
        inverted_index = None
        
        if index is not None:
            del index
        if enhanced_docs is not None:
            del enhanced_docs
        
        log(f"\n{'='*60}", "INFO")
        log("Qwen模型测试完成", "INFO")
        log(f"结果目录: {self.output_dir}", "INFO")
        log(f"{'='*60}", "INFO")
    
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
    
    def _generate_log_file(self, qwen_results):
        """生成日志文件"""
        log_content = f"""Qwen模型测试日志
        测试时间: {qwen_results['test_time']}
        总执行时间: {qwen_results.get('total_execution_time', 0):.2f}s
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
        
        log(f"✅ 日志文件已生成: {log_file}", "INFO")

# ====================== 主函数 ======================
def main():
    """主函数：运行Qwen模型测试"""
    
    log(f"{'='*60}", "INFO")
    log("Qwen模型测试", "INFO")
    log("测试场景：", "INFO")
    log("  1. 无提示词直接推理（测试学习能力）", "INFO")
    log("  2. 有提示词RAG推理（测试可训练能力）", "INFO")
    log(f"{'='*60}", "INFO")
    
    # 创建Qwen测试配置
    test_config = QwenTestConfig()
    
    # 创建测试运行器
    test_runner = QwenModelTestRunner(test_config)
    
    # 运行Qwen模型测试
    test_runner.run_qwen_tests()
    
    log(f"\n{'='*60}", "INFO")
    log("✅ Qwen模型测试完成", "INFO")
    log(f"{'='*60}", "INFO")

if __name__ == "__main__":
    main()