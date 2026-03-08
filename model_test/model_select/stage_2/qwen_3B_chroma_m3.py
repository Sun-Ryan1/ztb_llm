import torch
import json
import numpy as np
import chromadb
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
    """日志记录函数
"""
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
            "llm_name": "Qwen2.5-3B-Instruct",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct"
        }
        
        # 候选Embedding模型配置（第二阶段测试计划要求）
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
    def load_llm_model(config):
        """加载大语言模型
        返回: (tokenizer, llm_model)
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
            
            # 2. 加载大模型
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
            return tokenizer, llm_model
        except Exception as e:
            log(f"❌ Qwen模型加载失败，错误信息：{e}", "ERROR")
            return None, None
    
    @staticmethod
    def load_embedding_model(embedding_config):
        """加载Embedding模型
        参数: embedding_config
"""
        model_name = embedding_config["name"]
        local_path = embedding_config["local_path"]
        
        log(f"正在加载Embedding模型：{model_name}", "INFO")
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=True  # 使用快速tokenizer
            )
            embedding_model = AutoModel.from_pretrained(
                local_path,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            embedding_model.eval()
            log(f"✅ Embedding模型 {model_name} 加载成功", "INFO")
            return embedding_tokenizer, embedding_model
        except Exception as e:
            log(f"❌ Embedding模型加载失败，错误信息：{e}", "ERROR")
            return None, None
    
    @staticmethod
    def cleanup_models(llm_model, embedding_models=None):
        """清理模型资源
"""
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
    
    @staticmethod
    def cleanup_embedding_model(embedding_models):
        """清理Embedding模型资源
"""
        try:
            if embedding_models is not None:
                embedding_tokenizer, embedding_model = embedding_models
                del embedding_tokenizer
                del embedding_model
            
            torch.cuda.empty_cache()
            gc.collect()
            log("✅ Embedding模型资源已清理", "INFO")
        except Exception as e:
            log(f"⚠️  Embedding模型资源清理时发生错误：{e}", "WARNING")
    
    @staticmethod
    def cleanup_llm_model(tokenizer, llm_model):
        """清理大语言模型资源
"""
        try:
            if tokenizer is not None:
                del tokenizer
            if llm_model is not None:
                del llm_model
            
            torch.cuda.empty_cache()
            gc.collect()
            log("✅ 大语言模型资源已清理", "INFO")
        except Exception as e:
            log(f"⚠️  大语言模型资源清理时发生错误：{e}", "WARNING")

# ====================== BGE文本转向量函数 ======================
def bge_embedding_encode(embedding_models, text: Union[str, List[str]], batch_mode: bool = False) -> np.ndarray:
    """BGE文本转向量函数
    
    Args:
        _models: _tokenizer和embedding_model的元组
        text: 
        _mode: 是否使用批量模式
    
    Returns:
        .ndarray: 文本的向量表示
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

# ====================== 构建ChromaDB向量索引 ======================
def build_vector_index(embedding_models, docs: List[str], batch_size: int = 32) -> Tuple[Optional[chromadb.Collection], Optional[List[str]]]:
    """构建ChromaDB向量索引
    
    Args:
        _models: _tokenizer和embedding_model的元组
        docs: 
        _size: 批处理大小
    
    Returns:
        [Optional[chromadb.Collection], Optional[List[str]]]: ChromaDB集合和有效文档列表
"""
    valid_docs = []
    
    total_docs = len(docs)
    log(f"正在编码 {total_docs} 条文档...", "INFO")
    
    # 过滤空文档
    docs = [doc for doc in docs if doc.strip()]
    if not docs:
        log("❌ 无有效文档，无法构建ChromaDB索引", "ERROR")
        return None, None
    
    start_time = time.time()
    
    # 初始化ChromaDB客户端（使用内存模式，避免磁盘I/O错误）
    client = chromadb.EphemeralClient()
    
    # 创建或获取集合
    collection_name = "knowledge_base"
    try:
        # 如果集合存在则删除，重新创建
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "招投标知识库"},
        embedding_function=None  # 使用自定义embedding函数
    )
    
    # 批量处理文档
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_vectors = bge_embedding_encode(embedding_models, batch_docs, batch_mode=True)
        
        # 批量添加，避免循环
        if batch_vectors.size > 0:
            # 生成文档ID
            ids = [f"doc_{i+j}" for j in range(len(batch_docs))]
            
            # 向ChromaDB添加文档和向量
            collection.add(
                documents=batch_docs,
                embeddings=batch_vectors.tolist(),
                ids=ids,
                metadatas=[{"source": "knowledge_base"} for _ in batch_docs]
            )
            valid_docs.extend(batch_docs)
        
        # 日志输出，减少打印频率
        if (i // batch_size + 1) % 20 == 0:
            elapsed_time = time.time()
"""构建倒排索引，用于加速关键词检索
    
    Args:
        : 文档列表
"""global inverted_index
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

def enhanced_retrieval(embedding_models, collection, docs: List[str], question: str, top_k: int = 5, similarity_threshold: float = 0.75) -> List[str]:
    """改进的检索函数
    Args:
        _models: _tokenizer和embedding_model的元组
        collection: 
        : 文档列表
        question: 
        _k: 返回的文档数量
        similarity_threshold: 

    :
        List[str]: 检索到的文档列表
"""
    results = []
    
    # 1. 向量检索
"""关键词检索（后备）
    
    Args:
        : 查询问题
        docs: 
    
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
    """提取公司名
"""
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
    """提取关键词
"""
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
def direct_inference_no_prompt(tokenizer, llm_model, question, max_new_tokens=200, temperature=0.1, top_p=0.9, do_sample=True):
    """场景1：无任何提示词，直接让模型回答问题
    
    Args:
        : 模型tokenizer
        llm_model: 
        question: 查询问题
        max_new_tokens: 
        temperature: 生成温度
        top_p: 
        do_sample: 是否使用采样生成
    
    Returns:
        [str, List]: 模型回答和检索文档列表（此处为空）
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
        : 模型tokenizer
        llm_model: 
        questions: 查询问题列表
        max_new_tokens: 
        temperature: 生成温度
        top_p: 
        do_sample: 是否使用采样解码
    
    Returns:
        [Tuple[str, List]]: 模型回答和检索文档列表（此处为空）的列表
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

def optimized_rag_inference(tokenizer, llm_model, embedding_models, collection, docs, question, top_k=5, similarity_threshold=0.75, max_new_tokens=300, temperature=0.1, top_p=0.9, do_sample=False):
    """场景2：使用专业提示词模板的RAG推理
    
    Args:
        : 模型tokenizer
        llm_model: 
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        collection: 
        docs: 文档列表
        question: 
        top_k: 检索文档数量
        similarity_threshold: 
        max_new_tokens: 最大生成token数
        temperature: 
        top_p: 核采样参数
        do_sample: 
    
    Returns:
        Tuple[str, List[str]]: 模型回答和检索文档列表
"""
    retrieved_docs = enhanced_retrieval(embedding_models, collection, docs, question, top_k=top_k, similarity_threshold=similarity_threshold)
    
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
答："""inputs = tokenizer(
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

def batch_rag_inference(tokenizer, llm_model, embedding_models, collection, docs, questions, top_k=5, similarity_threshold=0.75, max_new_tokens=300, temperature=0.1, top_p=0.9, do_sample=False):
"""批量进行RAG推理
    
    Args:
        : 模型tokenizer
        llm_model: 
        embedding_models: 包含embedding_tokenizer和embedding_model的元组
        collection: 
        docs: 文档列表
        questions: 
        top_k: 检索文档数量
        similarity_threshold: 
        max_new_tokens: 最大生成token数
        temperature: 
        top_p: 核采样参数
        do_sample: 
    
    Returns:
        List[Tuple[str, List[str]]]: 模型回答和检索文档列表的列表
    """if not questions:
        return []
    
    # 1. 批量检索所有问题的相关文档
    all_retrieved_docs = []
    for question in questions:
        retrieved_docs = enhanced_retrieval(embedding_models, collection, docs, question, 
                                           top_k=top_k, similarity_threshold=similarity_threshold)
        all_retrieved_docs.append(retrieved_docs)
    
    # 2. 构建所有问题的prompt
    prompts = []
    for question, retrieved_docs in zip(questions, all_retrieved_docs):
        if not retrieved_docs:
            prompt = f
"""# 角色定位
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
答："""else:
            context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
            prompt = f
"""# 角色定位
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
答："""prompts.append(prompt)
    
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
        _file_path: 
        kb_file_path: 
    
    Returns:
        Tuple[List[Dict], List[str]]: 测试用例列表和知识库文档列表
    """try:
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
        : 输入文本
    
    Returns:
        [str]: 实体列表
    """entities = set()  # 使用set避免重复
    
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
        1: 1
        doc2: 2
    
    Returns:
        : 是否相关
    """entities1 = extract_entities_from_text(doc1)
    entities2 = extract_entities_from_text(doc2)
    
    # 检查是否有共同实体
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    # 使用SequenceMatcher计算相似度
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, doc1, doc2).ratio()
    return similarity > 0.6

def calculate_recall(retrieved_docs, relevant_docs):
"""计算召回率（默认返回是否至少召回一个相关文档）
    
    Args:
        _docs: 
        relevant_docs: 
    
    Returns:
        float: （1.0表示至少召回一个相关文档，0.0表示未召回）
    """if not retrieved_docs or not relevant_docs:
        return 0.0
    
    for retrieved_doc in retrieved_docs:
        for relevant_doc in relevant_docs:
            if is_doc_related(retrieved_doc, relevant_doc):
                return 1.0
    
    return 0.0

def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
"""计算Recall@k
    
    Args:
        _docs: 
        relevant_docs: 
        k: 
    
    Returns:
        : Recall@k值（0-1之间）
    """if not retrieved_docs or not relevant_docs:
        return 0.0
    
    # 只考虑前k个检索结果
    retrieved_at_k = retrieved_docs[:k]
    
    # 计算相关文档中被召回的数量
    relevant_found = 0
    for relevant_doc in relevant_docs:
        for retrieved_doc in retrieved_at_k:
            if is_doc_related(retrieved_doc, relevant_doc):
                relevant_found += 1
                break  # 每个相关文档只计数一次
    
    # Recall@k = 召回的相关文档数 / 总相关文档数
    return relevant_found / len(relevant_docs)

def calculate_multi_recall_at_k(retrieved_docs, relevant_docs, k_values):
"""计算多个k值的召回率
    
    Args:
        _docs: 
        relevant_docs: 
        k_values: 
    
    Returns:
        [int, float]: 不同k值对应的召回率
    """recall_results = {}
    for k in k_values:
        recall_results[k] = calculate_recall_at_k(retrieved_docs, relevant_docs, k)
    return recall_results

def calculate_accuracy(model_answer, reference_answer, threshold=0.6):
"""计算准确率
    
    Args:
        _answer: 
        reference_answer: 
        threshold: 
    
    Returns:
        : 准确率
    """if not model_answer or not reference_answer:
        return 0.0
    
    # 简单匹配检查
    if reference_answer in model_answer or model_answer in reference_answer:
        return 1.0
    
    # 关键词匹配
    match_count = 0
    total_keywords = 0
    
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in reference_answer:
            total_keywords += 1
            if keyword in model_answer:
                match_count += 1
    
    if total_keywords > 0:
        keyword_ratio = match_count / total_keywords
        if keyword_ratio >= threshold:
            return 1.0
    
    # 实体匹配
    ref_entities = extract_entities_from_text(reference_answer)
    model_entities = extract_entities_from_text(model_answer)
    
    if ref_entities:
        common_entities = set(ref_entities) & set(model_entities)
        if len(common_entities) / len(ref_entities) >= 0.5:
            return 1.0
    
    # 文本相似度匹配
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    return 1.0 if similarity > threshold else 0.0

def calculate_answer_quality(model_answer, reference_answer):
"""
    评估回答质量：包括相关性、完整性、一致性
    返回一个综合质量分数（0-1）
    
    Args:
        _answer: 
        reference_answer: 
    
    Returns:
        Dict: 
    """from difflib import SequenceMatcher
    
    # 1. 计算文本相似度
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    
    # 2. 检查是否包含关键信息
    keyword_hit = 0
    for keyword in IMPORTANT_KEYWORDS:
        if keyword in reference_answer and keyword in model_answer:
            keyword_hit += 1
    
    keyword_score = keyword_hit / len(IMPORTANT_KEYWORDS) if IMPORTANT_KEYWORDS else 0
    
    # 3. 检查回答格式
    format_score = 1.0
    for prefix in ERROR_PREFIXES:
        if model_answer.startswith(prefix):
            format_score -= 0.2
            break  # 只扣一次分
    
    # 4. 计算综合分数
    final_score = (similarity * 0.4) + (keyword_score * 0.4) + (format_score * 0.2)
    
    return {
        "quality_score": final_score,
        "similarity": similarity,
        "keyword_score": keyword_score,
        "format_score": format_score
    }

# ====================== 测试运行器 ======================
class QwenModelTestRunner:
"""Qwen模型测试运行器"""def __init__(self, config):
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
        """运行Qwen模型测试
"""
        log(f"\n{'='*80}", "INFO")
        log(f"开始第二阶段Embedding模型召回率测试", "INFO")
        log(f"大模型: {self.config.llm_config['llm_name']}", "INFO")
        log(f"测试用例数: {self.config.max_test_cases}", "INFO")
        log(f"知识库文档数: {len(self.knowledge_docs)}", "INFO")
        log(f"{'='*80}", "INFO")
        
        # 记录整体测试开始时间
        overall_start_time = time.time()
        
        # 加载大语言模型（只加载一次，用于所有Embedding模型测试）
        tokenizer, llm_model = ModelManager.load_llm_model(self.config.llm_config)
        if llm_model is None:
            log(f"❌ 大模型加载失败，测试终止", "ERROR")
            return
        
        # 记录所有Embedding模型的测试结果
        all_embedding_results = {}
        
        # 迭代测试每个Embedding模型
        for model_key, embedding_config in self.config.embedding_models_config.items():
            log(f"\n{'='*60}", "INFO")
            log(f"开始测试Embedding模型: {embedding_config['name']}", "INFO")
            log(f"模型路径: {embedding_config['local_path']}", "INFO")
            log(f"{'='*60}", "INFO")
            
            # 加载当前Embedding模型
            embedding_models = ModelManager.load_embedding_model(embedding_config)
            if embedding_models is None:
                log(f"❌ Embedding模型 {embedding_config['name']} 加载失败，跳过测试", "ERROR")
                continue
            
            # 构建向量索引
            log("\n正在构建ChromaDB向量索引...", "INFO")
            collection, enhanced_docs = build_vector_index(embedding_models, self.knowledge_docs,
                                                      batch_size=self.config.batch_size)
            if collection is None:
                log("⚠️  向量索引构建失败，场景2将无法测试", "WARNING")
                ModelManager.cleanup_embedding_model(embedding_models)
                continue
            
            # 运行测试
            scenario1_results = []
            scenario2_results = []
            
            # 测试用例批量处理支持
            batch_size = self.config.batch_size
            total_test_cases = len(self.test_cases)
            
            # 记录当前Embedding模型的测试开始时间
            embedding_start_time = time.time()
            
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
                            log(f"❌ 单样本场景1测试失败（测试用例ID：{case_idx + 1}）：{e2}", "ERROR")
                            continue
                
                # 场景2：使用RAG推理 - 支持批量处理
                try:
                    log(f"  正在处理场景2（RAG推理）：批量 {idx//batch_size + 1}/{total_test_cases//batch_size + 1}", "INFO")
                    questions = [case["question"] for case in batch_test_cases]
                    
                    log(f"    开始批量RAG推理，问题数量：{len(questions)}", "INFO")
                    batch_rag_answers = batch_rag_inference(
                        tokenizer, llm_model, embedding_models, collection, enhanced_docs,
                        questions,
                        top_k=self.config.top_k_retrieval,
                        similarity_threshold=self.config.similarity_threshold,
                        max_new_tokens=self.config.rag_max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample
                    )
                    log(f"    批量RAG推理完成，获得 {len(batch_rag_answers)} 个回答", "INFO")
                    
                    # 处理批量结果
                    for i, (model_answer2, retrieved_docs) in enumerate(batch_rag_answers):
                        case_idx = idx + i
                        test_case = batch_test_cases[i]
                        reference_answer = test_case["reference_answer"]
                        relevant_docs = test_case["relevant_docs"]
                        
                        # 计算准确率
                        accuracy2 = calculate_accuracy(model_answer2, reference_answer)
                        # 计算回答质量
                        quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
                        # 计算召回率
                        recall_results = calculate_multi_recall_at_k(retrieved_docs, relevant_docs, self.config.recall_at_k_values)
                        
                        result2 = {
                            "scenario": "rag_inference",
                            "test_case_id": case_idx + 1,
                            "question": test_case["question"],
                            "reference_answer": reference_answer,
                            "model_answer": model_answer2,
                            "retrieved_docs": retrieved_docs,
                            "relevant_docs": relevant_docs,
                            "accuracy": accuracy2,
                            "answer_length": len(model_answer2),
                            "quality_score": quality_metrics2["quality_score"],
                            "similarity": quality_metrics2["similarity"],
                            "keyword_score": quality_metrics2["keyword_score"],
                            "recall_at_k": recall_results
                        }
                        scenario2_results.append(result2)
                    
                except Exception as e:
                    log(f"❌ 场景2批量测试失败：{e}", "ERROR")
                    # 回退到单样本处理
                    for i, test_case in enumerate(batch_test_cases):
                        case_idx = idx + i
                        try:
                            model_answer2, retrieved_docs = optimized_rag_inference(
                                tokenizer, llm_model, embedding_models, collection, enhanced_docs,
                                test_case["question"],
                                top_k=self.config.top_k_retrieval,
                                similarity_threshold=self.config.similarity_threshold,
                                max_new_tokens=self.config.rag_max_new_tokens,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                do_sample=self.config.do_sample
                            )
                            reference_answer = test_case["reference_answer"]
                            relevant_docs = test_case["relevant_docs"]
                            
                            # 计算准确率
                            accuracy2 = calculate_accuracy(model_answer2, reference_answer)
                            # 计算回答质量
                            quality_metrics2 = calculate_answer_quality(model_answer2, reference_answer)
                            # 计算召回率
                            recall_results = calculate_multi_recall_at_k(retrieved_docs, relevant_docs, self.config.recall_at_k_values)
                            
                            result2 = {
                                "scenario": "rag_inference",
                                "test_case_id": case_idx + 1,
                                "question": test_case["question"],
                                "reference_answer": reference_answer,
                                "model_answer": model_answer2,
                                "retrieved_docs": retrieved_docs,
                                "relevant_docs": relevant_docs,
                                "accuracy": accuracy2,
                                "answer_length": len(model_answer2),
                                "quality_score": quality_metrics2["quality_score"],
                                "similarity": quality_metrics2["similarity"],
                                "keyword_score": quality_metrics2["keyword_score"],
                                "recall_at_k": recall_results
                            }
                            scenario2_results.append(result2)
                        except Exception as e2:
                            log(f"❌ 单样本场景2测试失败（测试用例ID：{case_idx + 1}）：{e2}", "ERROR")
                            continue
            
            # 统计场景2的平均召回率
            if scenario2_results:
                # 初始化召回率总和
                recall_sums = {k: 0.0 for k in self.config.recall_at_k_values}
                total_cases = len(scenario2_results)
                
                # 计算每个k值的召回率总和
                for result in scenario2_results:
                    for k, recall_value in result["recall_at_k"].items():
                        recall_sums[k] += recall_value
                
                # 计算平均召回率
                avg_recalls = {k: recall_sums[k] / total_cases for k in recall_sums}
                
                # 记录召回率结果到日志
                log(f"\n{'='*60}", "INFO")
                log(f"Embedding模型 {embedding_config['name']} 召回率结果", "INFO")
                log(f"{'='*60}", "INFO")
                for k in sorted(self.config.recall_at_k_values):
                    log(f"Recall@{k}: {avg_recalls[k]:.4f}", "INFO")
                log(f"{'='*60}", "INFO")
            
            # 记录当前Embedding模型的测试结果
            embedding_results = {
                "model_name": embedding_config["name"],
                "scenario1_results": scenario1_results,
                "scenario2_results": scenario2_results,
                "test_duration": time.time() - embedding_start_time
            }
            all_embedding_results[model_key] = embedding_results
            
            # 清理当前Embedding模型资源
            ModelManager.cleanup_embedding_model(embedding_models)
        
        # 清理大语言模型资源
        ModelManager.cleanup_llm_model(tokenizer, llm_model)
        
        # 保存测试结果到文件
        for model_key, embedding_results in all_embedding_results.items():
            model_name = embedding_results["model_name"]
            
            # 保存场景1结果
            if embedding_results["scenario1_results"]:
                scenario1_file = os.path.join(self.output_dir, "llm_results", f"{model_key}_scenario1_results.json")
                with open(scenario1_file, "w", encoding="utf-8") as f:
                    json.dump(embedding_results["scenario1_results"], f, ensure_ascii=False, indent=2)
                log(f"✅ 场景1结果已保存到: {scenario1_file}", "INFO")
            
            # 保存场景2结果
            if embedding_results["scenario2_results"]:
                scenario2_file = os.path.join(self.output_dir, "llm_results", f"{model_key}_scenario2_results.json")
                with open(scenario2_file, "w", encoding="utf-8") as f:
                    json.dump(embedding_results["scenario2_results"], f, ensure_ascii=False, indent=2)
                log(f"✅ 场景2结果已保存到: {scenario2_file}", "INFO")
                
                # 保存召回率统计
                recall_stats = {}
                for k in self.config.recall_at_k_values:
                    recall_values = [result["recall_at_k"][k] for result in embedding_results["scenario2_results"]]
                    recall_stats[f"recall@{k}"] = {
                        "mean": sum(recall_values) / len(recall_values),
                        "min": min(recall_values),
                        "max": max(recall_values),
                        "std": np.std(recall_values) if len(recall_values) > 1 else 0.0
                    }
                
                recall_file = os.path.join(self.output_dir, "llm_results", f"{model_key}_recall_stats.json")
                with open(recall_file, "w", encoding="utf-8") as f:
                    json.dump(recall_stats, f, ensure_ascii=False, indent=2, default=str)
                log(f"✅ 召回率统计已保存到: {recall_file}", "INFO")
        
        # 记录整体测试结束时间
        overall_end_time = time.time()
        log(f"\n{'='*80}", "INFO")
        log(f"第二阶段Embedding模型召回率测试完成", "INFO")
        log(f"总测试用时: {overall_end_time - overall_start_time:.2f}秒", "INFO")
        log(f"{'='*80}", "INFO")

# ====================== 主函数 ======================
if __name__ == "__main__":
    # 创建测试配置
    config = QwenTestConfig()
    
    # 创建测试运行器
    test_runner = QwenModelTestRunner(config)
    
    # 运行测试
    test_runner.run_qwen_tests()