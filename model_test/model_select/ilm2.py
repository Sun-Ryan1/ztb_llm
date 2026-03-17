import torch
import json
import numpy as np
import faiss
import os
import re
import sys
import random
import time
from datetime import datetime
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
from torch.utils.tensorboard import SummaryWriter
from difflib import SequenceMatcher

# ====================== 1. 初始化output目录 ======================
def init_output_dir(test_mode="with_prompt"):
    """根据测试模式初始化输出目录"""
    if test_mode == "with_prompt":
        output_dir = "./output_internlm2.5_with_prompt"
    else:
        output_dir = "./output_internlm2.5_without_prompt"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_log_dir = os.path.join(output_dir, f"tb_logs_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    return output_dir, tb_log_dir, result_dir, log_dir, cache_dir, timestamp

# ====================== 2. 日志重定向 ======================
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

# ====================== 3. TensorBoard初始化 ======================
def init_tensorboard(tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"✅ TensorBoard日志已初始化，存储路径：{tb_log_dir}")
    return writer

def log_to_tensorboard(writer, step, metrics):
    if "precision" in metrics:
        writer.add_scalar("准确率-Precision/单条用例", metrics["precision"], step)
    if "recall" in metrics:
        writer.add_scalar("召回率-Recall/单条用例", metrics["recall"], step)
    if "f1_score" in metrics:
        writer.add_scalar("F1分数/单条用例", metrics["f1_score"], step)
    if "f2_score" in metrics:
        writer.add_scalar("F2分数/单条用例", metrics["f2_score"], step)
    if "answer_length" in metrics:
        writer.add_scalar("回答长度/模型生成", metrics["answer_length"], step)
    if "reference_length" in metrics:
        writer.add_scalar("回答长度/标准答案", metrics["reference_length"], step)
    if "retrieved_count" in metrics:
        writer.add_scalar("检索文档数", metrics["retrieved_count"], step)
    
    # 添加更多tensorboard日志
    if "generation_time" in metrics:
        writer.add_scalar("时间/生成时间", metrics["generation_time"], step)
    if "retrieval_time" in metrics:
        writer.add_scalar("时间/检索时间", metrics["retrieval_time"], step)
    if "total_time" in metrics:
        writer.add_scalar("时间/总时间", metrics["total_time"], step)

# ====================== 4. 配置量化参数 ======================
def get_quantization_config():
    """
    获取模型量化配置
    
    Returns:
        BitsAndBytesConfig: 4-bit量化配置实例
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# ====================== 5. 加载InternLM2.5模型 ======================
def load_internlm2_5_model(model_path, cache_dir=None):
    """
    加载InternLM2.5模型和对应的分词器
    
    Args:
        model_path: 模型文件路径
        cache_dir: 缓存目录路径，用于存储模型文件
    
    Returns:
        tuple: (tokenizer, model)，分词器和模型实例
    """
    print(f"正在加载InternLM2.5模型：{model_path}")
    
    try:
        # 设置模型缓存路径
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",  # 右侧填充，符合模型预期
            local_files_only=os.path.exists(model_path)  # 本地文件优先
        )
        
        # 设置padding token（模型训练时使用）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        
        print(f"✅ InternLM2.5 Tokenizer加载成功，pad_token: {tokenizer.pad_token}")
        
        # 加载量化后的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",  # 自动分配设备
            quantization_config=get_quantization_config(),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,  # 低内存使用模式
            local_files_only=os.path.exists(model_path)
        )
        
        # 设置为评估模式
        model.eval()
        print(f"✅ InternLM2.5模型加载成功")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ InternLM2.5模型加载失败，错误信息：{e}")
        import traceback
        traceback.print_exc()
        return None, None

# ====================== 6. 加载BGE Embedding模型 ======================
def load_bge_embedding_model(embedding_path, cache_dir=None):
    """
    加载BGE Embedding模型，用于文本向量化
    
    Args:
        embedding_path: 模型文件路径
        cache_dir: 缓存目录路径
    
    Returns:
        tuple: (tokenizer, model)，分词器和Embedding模型实例
    """
    print(f"正在加载BGE Embedding模型：{embedding_path}")
    
    try:
        # 设置缓存路径
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            embedding_path,
            trust_remote_code=True,
            local_files_only=os.path.exists(embedding_path)
        )
        
        # 加载Embedding模型
        model = AutoModel.from_pretrained(
            embedding_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=os.path.exists(embedding_path)
        )
        
        # 设置为评估模式
        model.eval()
        print(f"✅ BGE Embedding模型加载成功")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ BGE Embedding模型加载失败，错误信息：{e}")
        import traceback
        traceback.print_exc()
        return None, None

# ====================== 7. 文本向量化函数 ======================
def generate_text_embedding(embedding_models, text, batch_mode=False):
    """
    使用BGE模型将文本转换为向量表示
    
    Args:
        embedding_models: tuple，包含(embedding_tokenizer, embedding_model)
        text: str或list，要转换的文本或文本列表
        batch_mode: bool，是否使用批量处理
    
    Returns:
        np.ndarray: 文本的向量表示
    """
    # 解包模型组件
    tokenizer, model = embedding_models
    
    # 检查模型是否有效
    if tokenizer is None or model is None:
        return np.array([])
    
    # 处理输入文本
    if isinstance(text, list):
        # 过滤空文本
        text_list = [str(t).strip() for t in text if str(t).strip()]
        if not text_list:
            return np.array([])
    else:
        # 处理单条文本
        text = str(text).strip()
        if not text:
            return np.array([])
        text_list = [text]
    
    # 最大序列长度
    max_seq_length = 512
    
    try:
        if batch_mode and len(text_list) > 1:
            # 批量处理多条文本
            inputs = tokenizer(
                text_list,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_length
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用CLS位置的输出作为文本表示
                embeddings = outputs.last_hidden_state[:, 0]
                # 归一化向量
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            
            return embeddings.cpu().numpy()
        else:
            # 单条文本处理
            embeddings_list = []
            for text_item in text_list:
                inputs = tokenizer(
                    text_item,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # 提取CLS位置的输出
                    hidden_state = outputs.last_hidden_state[:, 0]
                    embedding = hidden_state.cpu().numpy().squeeze()
                    
                    # 归一化处理
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    embeddings_list.append(embedding)
            
            # 返回单个向量或向量数组
            return embeddings_list[0] if len(embeddings_list) == 1 else np.array(embeddings_list)
    
    except Exception as e:
        print(f"❌ 向量编码失败：{e}")
        return np.array([])

# ====================== 7. 分析问题类型 ======================
def analyze_question_type(question):
    """
    分析问题类型，返回对应的类型标签
    """
    question_lower = question.lower()
    
    # 法定代表人查询
    if any(keyword in question_lower for keyword in ["法定代表人", "法人", "法人代表", "负责人"]):
        return "company_legal_representative"
    
    # 价格查询
    elif any(keyword in question_lower for keyword in ["价格", "多少钱", "费用", "成本", "价格是", "价格为"]):
        return "price_info"
    
    # 供应商查询
    elif any(keyword in question_lower for keyword in ["供应商", "供应", "提供", "供货", "供应商是", "供应商为"]):
        return "supplier_info"
    
    # 采购方查询
    elif any(keyword in question_lower for keyword in ["采购方", "采购", "购买", "买方", "采购方是", "采购方为"]):
        return "buyer_info"
    
    # 中标方查询
    elif any(keyword in question_lower for keyword in ["中标", "中标供应商", "中标方", "中标公司", "中标单位"]):
        return "bid_winner_info"
    
    # 基本信息查询
    elif any(keyword in question_lower for keyword in ["基本信息", "公司信息", "是什么", "基本情况", "介绍"]):
        return "basic_info"
    
    # 项目信息查询
    elif any(keyword in question_lower for keyword in ["项目", "招标", "投标", "项目基本情况", "采购项目"]):
        return "project_info"
    
    # 产品信息查询
    elif any(keyword in question_lower for keyword in ["产品", "设备", "仪器", "系统", "服务"]):
        return "product_info"
    
    # 法规类型查询
    elif any(keyword in question_lower for keyword in ["法规", "条例", "法", "属于什么", "类型"]):
        return "regulation_type"
    
    else:
        return "general_info"

# ====================== 8. 实体提取函数 ======================
def extract_key_entities(text):
    """
    从文本中提取关键实体，包括公司名、产品名、项目名和地名
    
    Args:
        text: str，要提取实体的文本
    
    Returns:
        list: 提取的关键实体列表
    """
    entities = []
    
    # 1. 提取公司名称（完整名称）
    company_patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{2,}有限公司)',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{2,}公司)',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{2,}集团)',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m for m in matches if isinstance(m, str) and len(m) > 5])  # 过滤短公司名
    
    # 2. 如果没有提取到完整公司名，尝试提取公司关键词
    if not any("有限公司" in entity or "公司" in entity for entity in entities):
        company_keyword_patterns = [
            r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:的|是)',  # 匹配"XXX的"或"XXX是"前面的部分
            r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)',  # 匹配带后缀的公司名
        ]
        for pattern in company_keyword_patterns:
            matches = re.findall(pattern, text)
            entities.extend([m for m in matches if isinstance(m, str) and len(m) > 2])
    
    # 3. 提取产品/项目名称
    product_project_patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})(?:的价格|是多少|基本信息|介绍)',
        r'介绍([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})',
        r'产品名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})',
        r'项目名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})'
    ]
    
    for pattern in product_project_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m.strip() for m in matches if isinstance(m, str) and len(m.strip()) > 3])
    
    # 4. 提取地名
    location_matches = re.findall(r'([\u4e00-\u9fa5]{2,6}市|[\u4e00-\u9fa5]{2,6}省|[\u4e00-\u9fa5]{2,6}区|[\u4e00-\u9fa5]{2,6}县)', text)
    entities.extend(location_matches)
    
    # 5. 去重并过滤无效实体
    unique_entities = []
    invalid_entities = ["有限公司", "是谁", "的", "是", "请", "介绍", "一下", "？", "\?"]
    
    for entity in entities:
        entity = entity.strip()
        if entity not in unique_entities and entity not in invalid_entities and len(entity) > 1:
            unique_entities.append(entity)
    
    # 6. 特殊处理：如果实体列表为空，尝试提取关键短语
    if not unique_entities:
        # 移除常见疑问词和标点
        cleaned_text = re.sub(r'[请介绍一下的是谁\?？，。；：、\s]', '', text)
        if len(cleaned_text) > 2:
            unique_entities.append(cleaned_text)
    
    return unique_entities

# ====================== 9. 多路召回检索系统 ======================
class MultiPathRetrievalSystem:
    def __init__(self, embedding_models, docs):
        self.embedding_models = embedding_models
        self.docs = docs
        self.doc_vectors = None
        self.faiss_index = None
        self.build_vector_index()
        
        # 构建倒排索引
        self.inverted_index = self.build_inverted_index()
    
    def build_vector_index(self):
        """构建向量索引"""
        print("正在构建向量索引...")
        doc_vectors = []
        
        for i, doc in enumerate(self.docs):
            vec = generate_text_embedding(self.embedding_models, doc)
            if vec.size > 0:
                doc_vectors.append(vec)
            
            if (i + 1) % 100 == 0:
                print(f"  已编码 {i+1}/{len(self.docs)} 条文档")
        
        if not doc_vectors:
            print("❌ 无有效文档向量")
            return
        
        self.doc_vectors = np.array(doc_vectors, dtype=np.float32)
        vector_dim = self.doc_vectors.shape[1]
        
        # 使用内积相似度
        self.faiss_index = faiss.IndexFlatIP(vector_dim)
        self.faiss_index.add(self.doc_vectors)
        
        print(f"✅ 向量索引构建完成，维度：{vector_dim}，文档数：{len(doc_vectors)}")
    
    def build_inverted_index(self):
        """构建倒排索引（关键词->文档索引）"""
        print("正在构建倒排索引...")
        inverted_index = defaultdict(set)
        
        for idx, doc in enumerate(self.docs):
            # 提取关键词
            keywords = self.extract_keywords_from_doc(doc)
            for keyword in keywords:
                inverted_index[keyword].add(idx)
            
            if (idx + 1) % 500 == 0:
                print(f"  已处理 {idx+1}/{len(self.docs)} 条文档")
        
        print(f"✅ 倒排索引构建完成，关键词数：{len(inverted_index)}")
        return inverted_index
    
    def extract_keywords_from_doc(self, doc):
        """从文档中提取关键词"""
        keywords = []
        
        # 提取公司名
        company_matches = re.findall(r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)', doc)
        keywords.extend(company_matches)
        
        # 提取人名
        name_matches = re.findall(r'法定代表人是([\u4e00-\u9fa5]{2,4})', doc)
        keywords.extend(name_matches)
        
        # 提取产品/项目名
        product_matches = re.findall(r'产品名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{3,})', doc)
        keywords.extend(product_matches)
        
        project_matches = re.findall(r'项目名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{3,})', doc)
        keywords.extend(project_matches)
        
        # 提取价格
        price_matches = re.findall(r'价格为?(\d+\.?\d*)元', doc)
        keywords.extend([f"{price}元" for price in price_matches])
        
        # 去除空关键词
        keywords = [k for k in keywords if k and len(k) > 1]
        
        return list(set(keywords))
    
    def vector_retrieval(self, query, top_k=10, similarity_threshold=0.05):
        """向量检索 - 优化版本"""
        if self.faiss_index is None or self.doc_vectors is None:
            return []
        
        # 快速生成查询向量
        query_vector = generate_text_embedding(self.embedding_models, query)
        if query_vector.size == 0:
            return []
        
        # 减少搜索候选数量，提高速度
        search_k = min(top_k * 3, len(self.docs))  # 从5倍减少到3倍
        distances, indices = self.faiss_index.search(query_vector.reshape(1, -1), search_k)
        
        results = []
        # 只保留相似度较高的结果，提前过滤
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.docs) and dist > similarity_threshold:
                results.append({
                    "doc": self.docs[idx],
                    "score": float(dist),
                    "type": "vector",
                    "index": idx
                })
                # 如果已经收集到足够的结果，提前返回
                if len(results) >= top_k:
                    break
        
        return results
    
    def keyword_retrieval(self, query, top_k=10):
        """关键词检索 - 增强版"""
        if not self.inverted_index:
            return []
        
        # 提取查询中的关键词
        query_entities = extract_key_entities(query)
        
        # 计算文档得分
        doc_scores = defaultdict(float)
        
        for entity in query_entities:
            if entity in self.inverted_index:
                for doc_idx in self.inverted_index[entity]:
                    # 公司名完全匹配得分更高
                    if "有限公司" in entity or "公司" in entity:
                        # 公司名完全匹配给予更高权重
                        doc_scores[doc_idx] += 2.0
                    else:
                        doc_scores[doc_idx] += 1.0
        
        # 检查是否有完整的公司名，如果没有，尝试部分匹配
        if not any("有限公司" in entity or "公司" in entity for entity in query_entities):
            # 尝试从查询中提取可能的公司名关键词
            query_words = query.split()
            for word in query_words:
                if len(word) >= 2 and word in self.inverted_index:
                    for doc_idx in self.inverted_index[word]:
                        doc_scores[doc_idx] += 0.5  # 部分匹配得分较低
        
        # 转换为结果列表
        results = []
        for doc_idx, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k*3]:
            if doc_idx < len(self.docs):
                results.append({
                    "doc": self.docs[doc_idx],
                    "score": score,
                    "type": "keyword",
                    "index": doc_idx
                })
        
        return results
    
    def rule_based_retrieval(self, query, top_k=5):
        """基于规则的检索"""
        results = []
        question_type = analyze_question_type(query)
        
        # 根据问题类型匹配特定模式
        patterns = {
            "company_legal_representative": ["法定代表人是", "法人代表是"],
            "price_info": ["价格为", "价格是", "价格为?", "价格是?"],
            "supplier_info": ["供应商是", "供应商为", "由.*?提供", "供应"],
            "bid_winner_info": ["中标供应商为", "中标供应商是", "中标方为", "中标方是"],
            "buyer_info": ["采购方为", "采购方是", "采购人为", "采购人是"],
            "project_info": ["项目名称为", "项目名称是", "采购项目"],
            "product_info": ["产品名称为", "产品名称是", "属于.*?类别"],
            "basic_info": ["产品名称为", "供应商是", "由.*?提供"]
        }
        
        if question_type in patterns:
            target_patterns = patterns[question_type]
            for idx, doc in enumerate(self.docs):
                for pattern in target_patterns:
                    if re.search(pattern, doc):
                        results.append({
                            "doc": doc,
                            "score": 1.0,  # 规则匹配给最高分
                            "type": "rule",
                            "index": idx
                        })
                        break  # 一个文档匹配一个模式即可
                
                if len(results) >= top_k * 3:
                    break
        
        return results
    
    def hybrid_retrieval(self, query, top_k=10, weights=None):
        """
        混合检索：结合向量、关键词和规则检索
        """
        if weights is None:
            weights = {"vector": 0.4, "keyword": 0.3, "rule": 0.3}
        
        # 并行执行多种检索，增加候选数量
        vector_results = self.vector_retrieval(query, top_k=top_k*5)  # 增加向量检索候选数量
        keyword_results = self.keyword_retrieval(query, top_k=top_k*5)  # 增加关键词检索候选数量
        rule_results = self.rule_based_retrieval(query, top_k=top_k*3)  # 增加规则检索候选数量
        
        # 合并结果
        all_results = {}
        
        # 处理向量结果
        for result in vector_results:
            doc_idx = result["index"]
            if doc_idx not in all_results:
                all_results[doc_idx] = {
                    "doc": result["doc"],
                    "scores": defaultdict(float),
                    "types": set()
                }
            all_results[doc_idx]["scores"]["vector"] = result["score"]
            all_results[doc_idx]["types"].add("vector")
        
        # 处理关键词结果
        for result in keyword_results:
            doc_idx = result["index"]
            if doc_idx not in all_results:
                all_results[doc_idx] = {
                    "doc": result["doc"],
                    "scores": defaultdict(float),
                    "types": set()
                }
            all_results[doc_idx]["scores"]["keyword"] = result["score"]
            all_results[doc_idx]["types"].add("keyword")
        
        # 处理规则结果
        for result in rule_results:
            doc_idx = result["index"]
            if doc_idx not in all_results:
                all_results[doc_idx] = {
                    "doc": result["doc"],
                    "scores": defaultdict(float),
                    "types": set()
                }
            all_results[doc_idx]["scores"]["rule"] = result["score"]
            all_results[doc_idx]["types"].add("rule")
        
        # 计算加权分数
        scored_results = []
        for doc_idx, data in all_results.items():
            total_score = 0
            for score_type, weight in weights.items():
                if score_type in data["scores"]:
                    total_score += data["scores"][score_type] * weight
            
            # 类型多样性奖励，增加奖励权重
            type_count = len(data["types"])
            if type_count > 1:
                total_score *= (1.0 + 0.2 * (type_count - 1))  # 从0.1提高到0.2
            
            scored_results.append({
                "doc": data["doc"],
                "score": total_score,
                "types": list(data["types"]),
                "index": doc_idx
            })
        
        # 按分数排序
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 重排序：确保高相关性的文档在前
        reranked_results = self.rerank_results(query, scored_results[:top_k*3])  # 增加重排序候选数量
        
        # 确保返回足够数量的结果，如果不足则补充
        if len(reranked_results) < top_k:
            # 从原始文档中补充一些可能相关的文档
            for i, doc in enumerate(self.docs):
                if i not in all_results and len(reranked_results) < top_k:
                    reranked_results.append({
                        "doc": doc,
                        "score": 0.1,  # 给予较低分数
                        "types": ["fallback"],
                        "index": i
                    })
        
        return reranked_results[:top_k]
    
    def rerank_results(self, query, results):
        """重排序结果 - 增强精确匹配"""
        if not results:
            return []
        
        # 提取查询实体
        query_entities = extract_key_entities(query)
        
        for result in results:
            doc = result["doc"]
            
            # 实体匹配奖励
            entity_match_score = 0
            
            for entity in query_entities:
                if entity in doc:
                    # 公司名完全匹配给予最高奖励
                    if ("有限公司" in entity or "公司" in entity) and entity in doc:
                        entity_match_score += 3.0
                    # 人名匹配
                    elif "法定代表人是" in query and entity in doc and "法定代表人是" in doc:
                        entity_match_score += 2.0
                    # 其他实体匹配
                    else:
                        entity_match_score += 0.5
            
            # 检查是否有完整的公司名匹配
            full_company_match = False
            for entity in query_entities:
                if ("有限公司" in entity or "公司" in entity) and entity in doc:
                    full_company_match = True
                    break
            
            if full_company_match:
                entity_match_score += 5.0  # 完整的公司名匹配给予最高奖励
            
            # 问题类型匹配奖励
            question_type = analyze_question_type(query)
            doc_lower = doc.lower()
            
            if question_type == "company_legal_representative" and "法定代表人是" in doc_lower:
                # 如果文档中包含"法定代表人是"，给予额外奖励
                entity_match_score += 2.0
                
                # 检查是否匹配查询中的公司名
                for entity in query_entities:
                    if ("有限公司" in entity or "公司" in entity) and entity in doc:
                        entity_match_score += 3.0
            elif question_type == "price_info" and ("价格为" in doc_lower or "价格是" in doc_lower):
                entity_match_score += 1.0
            elif question_type == "supplier_info" and ("供应商是" in doc_lower or "由" in doc_lower and "提供" in doc_lower):
                entity_match_score += 1.0
            elif question_type == "bid_winner_info" and ("中标供应商为" in doc_lower or "中标供应商是" in doc_lower):
                entity_match_score += 1.0
            elif question_type == "buyer_info" and ("采购方为" in doc_lower or "采购方是" in doc_lower):
                entity_match_score += 1.0
            
            # 更新分数 - 增加实体匹配的权重
            result["score"] = result["score"] * 0.6 + entity_match_score * 0.4
        
        # 重新排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results

# ====================== 10. InternLM2.5 生成函数 ======================
def generate_answer(tokenizer, model, prompt, max_new_tokens=100):
    """
    安全地生成InternLM2.5回答，优化生成质量和准确性
    """
    try:
        # 使用简化的prompt格式，避免复杂的对话标记
        formatted_prompt = f"问题：{prompt}\n请直接回答，不要添加任何解释："
        
        # 编码输入 - 确保正确处理attention_mask
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,  # 不使用padding，因为只有一个样本
            return_attention_mask=True  # 确保返回attention_mask
        )
        
        # 确保所有张量的batch维度都是1
        for key in inputs:
            if inputs[key].dim() > 0:
                # 确保是2D张量 (batch_size, sequence_length)
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)  # 增加batch维度
                elif inputs[key].shape[0] != 1:
                    inputs[key] = inputs[key][:1]  # 只取第一个样本
        
        # 确保attention_mask正确，避免与eos_token冲突
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 确保input_ids和attention_mask的长度相同
        if input_ids.shape[1] != attention_mask.shape[1]:
            min_len = min(input_ids.shape[1], attention_mask.shape[1])
            input_ids = input_ids[:, :min_len]
            attention_mask = attention_mask[:, :min_len]
        
        # 将张量移动到GPU上
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        
        # 使用InternLM2.5支持的生成参数，移除不支持的参数
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(max_new_tokens, 80),  # 进一步限制生成长度
                do_sample=False,  # 不使用采样
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 适当的重复惩罚
                no_repeat_ngram_size=3,  # 适当的n-gram大小
                use_cache=False,  # 禁用缓存，避免内存问题
            )
        
        # 解码并清理回答
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "请直接回答，不要添加任何解释：" in response:
            response = response.split("请直接回答，不要添加任何解释：")[-1].strip()
        elif "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        # 进一步清理回答，只保留核心内容
        cleaned_response = response.strip()
        
        # 移除多余的解释和说明
        if "根据提供的信息" in cleaned_response:
            cleaned_response = cleaned_response.replace("根据提供的信息", "")
        if "这一结论直接来源于" in cleaned_response:
            cleaned_response = cleaned_response.split("这一结论直接来源于")[0]
        if "因此，可以确认" in cleaned_response:
            cleaned_response = cleaned_response.split("因此，可以确认")[0]
        if "综上所述" in cleaned_response:
            cleaned_response = cleaned_response.split("综上所述")[0]
        
        # 移除多余的标点符号和空格
        cleaned_response = re.sub(r'[，。；：、\s]+', '', cleaned_response)
        
        # 限制回答长度
        if len(cleaned_response) > 50:
            cleaned_response = cleaned_response[:50]
        
        # 如果回答为空，返回未知
        return cleaned_response if cleaned_response else "未知"
    
    except RuntimeError as e:
        if "dimension" in str(e).lower() or "size" in str(e).lower():
            print(f"⚠️ 检测到维度不匹配错误，使用简化方法...")
            return generate_answer_simple(tokenizer, model, prompt, max_new_tokens)
        else:
            print(f"❌ InternLM2.5生成失败: {e}")
            return "未知"
    
    except Exception as e:
        print(f"❌ InternLM2.5生成失败: {e}")
        return "未知"

def generate_answer_simple(tokenizer, model, prompt, max_new_tokens=100):
    """
    简化的回答生成方法，用于处理复杂生成失败的情况
    """
    try:
        # 最简单的prompt格式
        simple_prompt = f"问题：{prompt}\n回答："
        
        # 使用tokenizer()方法替代encode()，确保返回input_ids和attention_mask
        inputs = tokenizer(
            simple_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,  # 不使用padding，因为只有一个样本
            return_attention_mask=True
        )
        
        # 确保所有张量的batch维度都是1
        for key in inputs:
            if inputs[key].dim() > 0:
                # 确保是2D张量 (batch_size, sequence_length)
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)  # 增加batch维度
                elif inputs[key].shape[0] != 1:
                    inputs[key] = inputs[key][:1]  # 只取第一个样本
        
        # 确保input_ids和attention_mask的长度相同
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 确保input_ids和attention_mask的维度相同
        if input_ids.shape[1] != attention_mask.shape[1]:
            min_len = min(input_ids.shape[1], attention_mask.shape[1])
            input_ids = input_ids[:, :min_len]
            attention_mask = attention_mask[:, :min_len]
        
        # 将张量移动到GPU上
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        
        # 最小化的生成参数，移除不支持的temperature参数
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(max_new_tokens, 80),
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                use_cache=True
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除输入部分
        if response.startswith(simple_prompt):
            response = response[len(simple_prompt):].strip()
        
        return response if response else "未知"
    
    except Exception as e:
        print(f"❌ 简化生成方法失败: {e}")
        # 添加详细的调试信息
        try:
            print(f"  输入张量信息:")
            print(f"  input_ids形状: {input_ids.shape if 'input_ids' in locals() else '未定义'}")
            print(f"  attention_mask形状: {attention_mask.shape if 'attention_mask' in locals() else '未定义'}")
            print(f"  batch_size: {input_ids.shape[0] if 'input_ids' in locals() and input_ids.dim() > 0 else '未知'}")
        except Exception:
            pass
        return "未知"

def rag_inference(tokenizer, model, retrieval_system, question, test_mode="with_prompt", max_new_tokens=100):
    """
    InternLM2.5 RAG推理函数，结合检索和生成
    
    Args:
        tokenizer: 分词器实例
        model: 模型实例
        retrieval_system: 检索系统实例
        question: 用户问题
        test_mode: 测试模式，"with_prompt"或"without_prompt"
        max_new_tokens: 最大生成token数
    
    Returns:
        tuple: (生成的回答, 检索到的文档列表)
    """
    # 检索相关文档
    retrieved_items = retrieval_system.hybrid_retrieval(question, top_k=5)
    retrieved_docs = [item["doc"] for item in retrieved_items]
    
    # 处理无结果情况
    if not retrieved_docs:
        return "未知", []
    
    # 根据测试模式选择提示构建方式
    if test_mode == "with_prompt":
        prompt = get_prompt_with_template(question, retrieved_docs)
    else:
        prompt = get_prompt_without_template(question, retrieved_docs)
    
    # 生成回答
    model_answer = generate_answer(tokenizer, model, prompt, max_new_tokens)
    
    return model_answer, retrieved_docs

def get_prompt_with_template(question, retrieved_docs):
    """
    获取使用提示词模板的prompt（第二个调研方向）
    """
    # 分析问题类型
    question_type = analyze_question_type(question)
    
    # 构建上下文（限制文档数量和长度）
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:2]):  # 减少文档数量到2个，提高模型处理速度
        # 截断文档长度，只保留核心信息
        if len(doc) > 200:
            doc = doc[:200] + "..."
        context_parts.append(f"信息{i+1}: {doc}")
    
    context = "\n".join(context_parts)
    
    # 根据问题类型定制更明确的指令
    instruction_templates = {
        "company_legal_representative": "请直接从上下文中提取法定代表人的姓名，只回答姓名，不添加任何其他内容。如果找不到，回答'未知'。",
        "price_info": "请直接从上下文中提取价格数值，格式为'XX元'，只回答价格，不添加任何其他内容。如果找不到，回答'未知'。",
        "supplier_info": "请直接从上下文中提取供应商的完整名称，只回答公司名，不添加任何其他内容。如果找不到，回答'未知'。",
        "bid_winner_info": "请直接从上下文中提取中标供应商的完整名称，只回答公司名，不添加任何其他内容。如果找不到，回答'未知'。",
        "buyer_info": "请直接从上下文中提取采购方的完整名称，只回答单位名，不添加任何其他内容。如果找不到，回答'未知'。",
        "project_info": "请直接从上下文中提取项目名称、中标方、采购方、合同金额等核心信息，用简洁的语言拼接，不添加任何解释。如果找不到，回答'未知'。",
        "product_info": "请直接从上下文中提取产品名称、供应商、类别等核心信息，用简洁的语言拼接，不添加任何解释。如果找不到，回答'未知'。",
        "regulation_type": "请直接从上下文中提取法规类型，只回答'条例'或'法'等类型名称，不添加任何其他内容。如果找不到，回答'未知'。",
        "basic_info": "请直接从上下文中提取核心信息，只回答关键内容，不添加任何解释或额外信息。如果找不到，回答'未知'。",
        "general_info": "请直接从上下文中提取答案，只回答关键内容，不添加任何解释或额外信息。如果找不到，回答'未知'。"
    }
    
    instruction = instruction_templates.get(question_type, instruction_templates["general_info"])
    
    # 构建用户消息 - 更加简洁明确，强调直接提取答案
    user_message = f"""你是一个专业的信息提取助手，只能根据提供的上下文回答问题，严格遵循以下规则：
- 只提取上下文中明确提到的信息，不进行任何推理或扩展
- 答案必须简洁，不超过20个字
- 不要添加任何解释、说明或多余的文字
- 只回答问题所要求的内容，不要回答其他无关信息
- 如果上下文中没有相关信息，直接回答'未知'
- 如果上下文中明确提到'未知'，直接回答'未知'

上下文信息：
{context}

问题：{question}

{instruction}

答案："""
    
    return user_message

def get_prompt_without_template(question, retrieved_docs):
    """
    获取不使用提示词模板的prompt（第一个调研方向）
    """
    # 构建上下文（限制文档数量和长度）
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:2]):  # 最多2个文档
        # 截断文档长度
        if len(doc) > 300:
            doc = doc[:300] + "..."
        context_parts.append(doc)
    
    context = "\n".join(context_parts)
    
    # 简化的prompt，只提供上下文和问题，不添加任何指令
    prompt = f"""以下是相关信息：

{context}

问题：{question}

请根据以上信息回答问题："""
    
    return prompt

# ====================== 11. 加载问答对 ======================
def load_and_prepare_test_data(qa_file_path="qa_data/520_qa.json", kb_file_path="qa_data/knowledge_base.txt"):
    """
    加载和准备测试数据
    """
    print("正在加载测试数据...")
    
    try:
        # 1. 加载知识库文档
        knowledge_docs = []
        if os.path.exists(kb_file_path):
            with open(kb_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 10:  # 过滤太短的文档
                        knowledge_docs.append(line)
            
            print(f"✅ 加载 {len(knowledge_docs)} 条知识库文档")
        
        # 2. 加载问答对
        test_cases = []
        
        if os.path.exists(qa_file_path):
            with open(qa_file_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            
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
                
                # 如果没有相关文档，使用答案作为备选
                if not relevant_docs:
                    relevant_docs = [answer]
                
                test_cases.append({
                    "test_case_id": idx + 1,
                    "question": question,
                    "reference_answer": answer,
                    "relevant_docs": relevant_docs,
                    "question_type": analyze_question_type(question)
                })
            
            print(f"✅ 加载 {len(test_cases)} 条有效测试用例")
        
        else:
            print(f"❌ 未找到问答对文件: {qa_file_path}")
            # 创建示例测试用例
            test_cases = create_sample_test_cases()
            print(f"⚠️  创建 {len(test_cases)} 条示例测试用例")
        
        # 3. 如果知识库为空，从测试用例构建
        if not knowledge_docs:
            all_docs_set = set()
            for case in test_cases:
                for doc in case["relevant_docs"]:
                    all_docs_set.add(doc)
            knowledge_docs = list(all_docs_set)
            print(f"📚 从测试用例构建 {len(knowledge_docs)} 条知识库文档")
        
        # 4. 统计问题类型分布
        type_stats = defaultdict(int)
        for case in test_cases:
            type_stats[case["question_type"]] += 1
        
        print("\n📊 问题类型分布:")
        for q_type, count in sorted(type_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(test_cases) * 100
            print(f"  {q_type}: {count} 条 ({percentage:.1f}%)")
        
        return test_cases, knowledge_docs
    
    except Exception as e:
        print(f"❌ 加载测试数据失败: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def create_sample_test_cases():
    """创建示例测试用例（用于测试）"""
    return [
        {
            "test_case_id": 1,
            "question": "亳州朗朗生物科技有限公司的法定代表人是谁？",
            "reference_answer": "亳州朗朗生物科技有限公司的法定代表人是刘超",
            "relevant_docs": ["亳州朗朗生物科技有限公司的法定代表人是刘超，该公司注册于安徽亳州，具有合法的企业资质"],
            "question_type": "company_legal_representative"
        },
        {
            "test_case_id": 2,
            "question": "37平米网架卫生帐篷的价格是多少？",
            "reference_answer": "霸州市森汛安消防护用品提供的37平米网架卫生帐篷的价格为22900.0元",
            "relevant_docs": ["霸州市森汛安消防护用品提供的37平米网架卫生帐篷 网架式救灾援外帐篷 野外拱型救援帐篷的价格为22900.0元，该价格包含了产品的成本及合理利润"],
            "question_type": "price_info"
        }
    ]

# ====================== 12. 评估函数（包含F1和F2指标）======================
def calculate_enhanced_recall(retrieved_docs, relevant_docs):
    """
    增强版召回率计算
    """
    if not retrieved_docs or not relevant_docs:
        return 0
    
    # 检查每个相关文档是否在检索到的文档中（或高度相关）
    for relevant_doc in relevant_docs:
        for retrieved_doc in retrieved_docs:
            if is_document_match(retrieved_doc, relevant_doc):
                return 1
    
    return 0

def is_document_match(doc1, doc2):
    """
    判断两个文档是否匹配
    """
    # 直接包含关系
    if doc1 in doc2 or doc2 in doc1:
        return True
    
    # 提取关键实体比较
    entities1 = extract_key_entities(doc1)
    entities2 = extract_key_entities(doc2)
    
    # 如果有共同实体，认为匹配
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    # 文本相似度
    similarity = calculate_text_similarity(doc1, doc2)
    return similarity > 0.7

def extract_key_entities(text):
    """提取关键实体"""
    entities = []
    
    # 公司名
    company_matches = re.findall(r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)', text)
    entities.extend(company_matches)
    
    # 人名
    name_matches = re.findall(r'法定代表人是([\u4e00-\u9fa5]{2,4})', text)
    entities.extend(name_matches)
    
    # 产品/项目名
    product_matches = re.findall(r'产品名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{3,})', text)
    entities.extend(product_matches)
    
    project_matches = re.findall(r'项目名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{3,})', text)
    entities.extend(project_matches)
    
    # 价格
    price_matches = re.findall(r'价格为?(\d+\.?\d*)元', text)
    entities.extend([f"{price}元" for price in price_matches])
    
    return list(set(entities))

def calculate_text_similarity(text1, text2):
    """计算文本相似度"""
    # 简单实现：使用SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_enhanced_accuracy(model_answer, reference_answer, question):
    """
    增强版准确率计算 - 针对不同类型问题使用不同评估策略
    """
    if not model_answer or not reference_answer:
        return 0
    
    # 预处理
    model_answer_clean = preprocess_answer(model_answer)
    reference_answer_clean = preprocess_answer(reference_answer)
    
    # 1. 完全匹配
    if reference_answer == model_answer or reference_answer in model_answer or model_answer in reference_answer:
        return 1
    
    # 2. 清洗后匹配
    if reference_answer_clean == model_answer_clean:
        return 1
    
    # 3. 处理"未知"情况
    if is_unknown_answer(reference_answer):
        if is_unknown_answer(model_answer):
            return 1
    
    # 4. 根据问题类型使用不同的评估策略
    question_type = analyze_question_type(question)
    
    if question_type == "basic_info":
        # 对于基本信息类问题，使用更宽松的评估
        return calculate_basic_info_accuracy(model_answer, reference_answer)
    elif question_type == "company_legal_representative":
        # 对于法定代表人问题，比较人名
        model_name = extract_name(model_answer)
        ref_name = extract_name(reference_answer)
        # 提取参考答案中的准确人名（确保只提取法定代表人姓名）
        ref_name_from_answer = re.search(r'法定代表人是([\u4e00-\u9fa5]{2,4})', reference_answer)
        if ref_name_from_answer:
            ref_name = ref_name_from_answer.group(1)
        
        if model_name and ref_name and model_name == ref_name:
            return 1
    elif question_type == "price_info":
        # 对于价格问题，比较价格数字
        model_price = extract_price(model_answer)
        ref_price = extract_price(reference_answer)
        if model_price and ref_price:
            # 处理可能的小数位数差异
            if float(model_price) == float(ref_price):
                return 1
    elif question_type == "supplier_info":
        # 对于供应商问题，比较供应商名称
        model_supplier = extract_supplier(model_answer)
        ref_supplier = extract_supplier(reference_answer)
        # 从参考答案中提取准确供应商
        ref_supplier_from_answer = re.search(r'供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})|由([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供', reference_answer)
        if ref_supplier_from_answer:
            ref_supplier = ref_supplier_from_answer.group(1) or ref_supplier_from_answer.group(2)
        
        if model_supplier and ref_supplier:
            # 宽松匹配：检查核心关键词是否匹配
            model_core = re.sub(r'[省市县区有限公司集团]', '', model_supplier)
            ref_core = re.sub(r'[省市县区有限公司集团]', '', ref_supplier)
            if model_core in ref_core or ref_core in model_core:
                return 1
    elif question_type == "buyer_info":
        # 对于采购方问题，比较采购方名称
        model_buyer = extract_buyer(model_answer)
        ref_buyer = extract_buyer(reference_answer)
        # 从参考答案中提取准确采购方
        ref_buyer_from_answer = re.search(r'采购方为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})|采购方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})', reference_answer)
        if ref_buyer_from_answer:
            ref_buyer = ref_buyer_from_answer.group(1) or ref_buyer_from_answer.group(2)
        
        if model_buyer and ref_buyer:
            # 宽松匹配：检查核心关键词是否匹配
            model_core = re.sub(r'[省市县区有限公司集团]', '', model_buyer)
            ref_core = re.sub(r'[省市县区有限公司集团]', '', ref_buyer)
            if model_core in ref_core or ref_core in model_core:
                return 1
    elif question_type == "bid_winner_info":
        # 对于中标方问题，比较中标方名称
        model_winner = extract_winner(model_answer)
        ref_winner = extract_winner(reference_answer)
        # 从参考答案中提取准确中标方
        ref_winner_from_answer = re.search(r'中标供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})|中标供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})', reference_answer)
        if ref_winner_from_answer:
            ref_winner = ref_winner_from_answer.group(1) or ref_winner_from_answer.group(2)
        
        if model_winner and ref_winner:
            # 宽松匹配：检查核心关键词是否匹配
            model_core = re.sub(r'[省市县区有限公司集团]', '', model_winner)
            ref_core = re.sub(r'[省市县区有限公司集团]', '', ref_winner)
            if model_core in ref_core or ref_core in model_core:
                return 1
    elif question_type == "regulation_type":
        # 对于法规类型问题，直接匹配类型
        if "条例" in reference_answer and "条例" in model_answer:
            return 1
        if "法" in reference_answer and "法" in model_answer:
            return 1
    
    # 5. 提高相似度阈值的灵敏度，特别是对于短文本
    similarity_threshold = 0.6 if question_type == "basic_info" else 0.7
    similarity = calculate_text_similarity(model_answer_clean, reference_answer_clean)
    if similarity > similarity_threshold:
        return 1
    
    return 0

def calculate_basic_info_accuracy(model_answer, reference_answer):
    """
    计算基本信息类问题的准确率（更宽松的评估）
    """
    # 提取关键信息进行比较
    key_info_pairs = [
        # (模型提取函数, 参考提取函数, 权重)
        (extract_product_name, extract_product_name, 0.3),
        (extract_company_name, extract_company_name, 0.3),
        (extract_product_category, extract_product_category, 0.2),
        (extract_price, extract_price, 0.1),
        (extract_standard_info, extract_standard_info, 0.1),
    ]
    
    total_score = 0
    total_weight = 0
    
    for model_extract_func, ref_extract_func, weight in key_info_pairs:
        model_info = model_extract_func(model_answer)
        ref_info = ref_extract_func(reference_answer)
        
        if model_info and ref_info:
            # 检查信息是否匹配
            if is_info_match(model_info, ref_info):
                total_score += weight
        
        total_weight += weight
    
    # 如果匹配度超过60%，认为正确
    accuracy_score = total_score / total_weight if total_weight > 0 else 0
    return 1 if accuracy_score > 0.6 else 0

def extract_product_name(text):
    """提取产品名称"""
    patterns = [
        r'产品名称为([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})是一款',
        r'介绍([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-\s]{3,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # 尝试提取可能的产品型号
    model_match = re.search(r'([A-Za-z]{2,}\s*\d{3,})', text)
    if model_match:
        return model_match.group(1)
    
    return None

def extract_company_name(text):
    """提取公司名称"""
    patterns = [
        r'由([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供',
        r'供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})供应',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,}公司)',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,}科技有限公司)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return None

def extract_product_category(text):
    """提取产品类别"""
    categories = ["其他类别", "仪器仪表", "测试仪器", "水质测试", "电导率仪表"]
    
    for category in categories:
        if category in text:
            return category
    
    return None

def extract_standard_info(text):
    """提取标准信息"""
    if "符合" in text and ("标准" in text or "规格" in text):
        # 简化处理：只要有"符合标准"相关表述
        return "符合标准"
    return None

def is_info_match(info1, info2):
    """
    判断两个信息是否匹配（允许部分匹配）
    """
    if not info1 or not info2:
        return False
    
    # 完全匹配
    if info1 == info2:
        return True
    
    # 互相包含
    if info1 in info2 or info2 in info1:
        return True
    
    # 对于公司名，允许简写匹配
    if ("科技" in info1 and "科技" in info2) or ("公司" in info1 and "公司" in info2):
        # 提取核心名称部分
        core1 = re.sub(r'[省市县区]', '', info1)
        core2 = re.sub(r'[省市县区]', '', info2)
        
        # 检查是否有共同的关键词
        words1 = set(re.findall(r'[\u4e00-\u9fa5]{2,}', core1))
        words2 = set(re.findall(r'[\u4e00-\u9fa5]{2,}', core2))
        
        common_words = words1 & words2
        if len(common_words) >= 2:  # 至少有2个共同关键词
            return True
    
    # 文本相似度
    similarity = calculate_text_similarity(info1, info2)
    return similarity > 0.7

def preprocess_answer(answer):
    """预处理答案"""
    if not answer:
        return ""
    
    # 移除标点符号和空格
    answer = re.sub(r'[，。；：、\s]', '', answer)
    
    # 转换为小写（对于中文影响不大，但处理可能存在的英文）
    answer = answer.lower()
    
    return answer

def is_unknown_answer(answer):
    """判断是否为未知答案"""
    unknown_keywords = ["未知", "无法确定", "不能确定", "不清楚", "不明确", "暂无", "没有", "无"]
    return any(keyword in answer for keyword in unknown_keywords)

def extract_name(text):
    """提取人名 - 优化版"""
    # 针对法定代表人问题，优先匹配明确的人名模式
    name_patterns = [
        r'法定代表人是([\u4e00-\u9fa5]{2,4})',
        r'法人是([\u4e00-\u9fa5]{2,4})',
        r'法人代表是([\u4e00-\u9fa5]{2,4})',
        r'代表人为([\u4e00-\u9fa5]{2,4})',
        r'负责人是([\u4e00-\u9fa5]{2,4})',
        r'([\u4e00-\u9fa5]{2,4})是法定代表人',
        r'([\u4e00-\u9fa5]{2,4})是法人',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) if match.group(1) else match.group(2)
    
    # 如果没有明确的法定代表人模式，尝试提取文本中的人名
    # 只提取2-4个汉字的人名，排除常见错误
    match = re.search(r'([\u4e00-\u9fa5]{2,4})', text)
    if match:
        name = match.group(1)
        # 排除明显不是人名的情况
        if name not in ["有限公司", "公司", "集团", "未知", "无法", "不能", "清楚", "明确", "暂无", "没有", "无"]:
            return name
    
    return None

def extract_price(text):
    """提取价格"""
    match = re.search(r'(\d+\.?\d*)元', text)
    return match.group(1) if match else None

def extract_supplier(text):
    """提取供应商"""
    # 尝试多种模式
    patterns = [
        r'供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'由([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def extract_winner(text):
    """提取中标方"""
    patterns = [
        r'中标供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'中标供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'中标方为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'中标方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

def extract_buyer(text):
    """提取采购方"""
    patterns = [
        r'采购方为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'采购方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'采购人为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'采购人是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return None

# ====================== 12b. F1和F2指标计算函数 ======================
def calculate_f1_score(precision, recall):
    """计算F1分数"""
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def calculate_f2_score(precision, recall):
    """计算F2分数（更重视召回率）"""
    if precision + recall == 0:
        return 0
    return 5 * precision * recall / (4 * precision + recall)

def calculate_all_metrics(model_answer, reference_answer, retrieved_docs, relevant_docs, question):
    """
    计算所有评估指标
    返回：precision, recall, f1, f2
    """
    # 计算准确率（作为precision）
    precision = calculate_enhanced_accuracy(model_answer, reference_answer, question)
    
    # 计算召回率
    recall = calculate_enhanced_recall(retrieved_docs, relevant_docs)
    
    # 计算F1和F2
    f1 = calculate_f1_score(precision, recall)
    f2 = calculate_f2_score(precision, recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2
    }

# ====================== 13. 主测试流程 ======================
def run_optimized_internlm2_5_test(test_mode="with_prompt"):
    """
    优化的InternLM2.5模型测试流程
    test_mode: 
        - "with_prompt": 使用提示词模板（第二个调研方向）
        - "without_prompt": 不使用提示词模板（第一个调研方向）
    """
    # 根据测试模式初始化
    if test_mode == "with_prompt":
        print("="*60)
        print("模式：使用提示词模板（第二个调研方向）")
        print("描述：构建提示词模板对大模型加以训练，以此方式来告诉大模型该如何回答问题")
        print("="*60)
    else:
        print("="*60)
        print("模式：不使用提示词模板（第一个调研方向）")
        print("描述：不做任何处理情况下，观察各模型的回答情况")
        print("="*60)
    
    output_dir, tb_log_dir, result_dir, log_dir, cache_dir, timestamp = init_output_dir(test_mode)
    log_file_path = os.path.join(log_dir, f"internlm2_5_run_log_{timestamp}.txt")
    sys.stdout = LoggerRedirect(log_file_path)
    tb_writer = init_tensorboard(tb_log_dir)
    
    # 模型配置
    test_config = {
        "llm_name": "internlm2_5-7b-chat",
        "model_local_path": "/mnt/workspace/data/modelscope/cache/Shanghai_AI_Laboratory/internlm2_5-7b-chat",
        "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5/BAAI/bge-large-zh-v1___5"
    }
    
    # 1. 加载测试数据
    print("\n📂 加载测试数据...")
    qa_file_path = "qa_data/520_qa.json"
    kb_file_path = "qa_data/knowledge_base.txt"
    
    test_cases, knowledge_docs = load_and_prepare_test_data(qa_file_path, kb_file_path)
    
    if not test_cases:
        print("❌ 无测试用例，测试终止")
        tb_writer.close()
        return
    
    print(f"\n✅ 数据加载完成:")
    print(f"   测试用例: {len(test_cases)} 条")
    print(f"   知识库文档: {len(knowledge_docs)} 条")
    
    # 2. 加载InternLM2.5模型
    print("\n🤖 加载InternLM2.5模型...")
    tokenizer, model = load_internlm2_5_model(test_config["model_local_path"], cache_dir)
    
    if tokenizer is None or model is None:
        print("❌ 模型加载失败，测试终止")
        tb_writer.close()
        return
    
    # 3. 加载BGE Embedding模型
    print("\n🔍 加载BGE Embedding模型...")
    embedding_models = load_bge_embedding_model(test_config["embedding_local_path"], cache_dir)
    
    if embedding_models is None:
        print("❌ Embedding模型加载失败，测试终止")
        tb_writer.close()
        return
    
    # 4. 构建多路召回检索系统
    print("\n🔧 构建多路召回检索系统...")
    retrieval_system = MultiPathRetrievalSystem(embedding_models, knowledge_docs)
    
    if retrieval_system.faiss_index is None:
        print("❌ 检索系统构建失败，测试终止")
        tb_writer.close()
        return
    
    # 5. 执行测试
    print("\n" + "="*60)
    print("开始执行测试...")
    print("="*60)
    
    test_results = []
    successful_generations = 0
    failed_generations = 0
    
    for idx, case in enumerate(test_cases):
        question = case["question"]
        reference_answer = case["reference_answer"]
        relevant_docs = case["relevant_docs"]
        question_type = case["question_type"]
        
        print(f"\n--- 测试用例 {idx+1}/{len(test_cases)} [{question_type}] ---")
        print(f"问题: {question[:80]}..." if len(question) > 80 else f"问题: {question}")
        
        # 调试：显示提取的实体
        query_entities = extract_key_entities(question)
        print(f"📝 提取的实体: {query_entities}")
        
        # 记录时间
        start_time = time.time()
        retrieval_start = time.time()
        
        # 执行推理（根据测试模式选择不同的推理方式）
        model_answer, retrieved_docs = rag_inference(
            tokenizer, model, retrieval_system, question, test_mode
        )
        
        # 调试：显示检索到的文档
        print(f"🔍 检索到 {len(retrieved_docs)} 个文档:")
        for i, doc in enumerate(retrieved_docs[:3]):  # 只显示前3个
            print(f"  文档{i+1}: {doc[:100]}...")
        
        retrieval_time = time.time() - retrieval_start
        generation_time = time.time() - start_time - retrieval_time
        total_time = time.time() - start_time
        
        # 统计生成成功/失败
        if model_answer != "未知" and "生成失败" not in model_answer:
            successful_generations += 1
        else:
            failed_generations += 1
        
        # 计算所有指标（包含F1和F2）
        metrics = calculate_all_metrics(model_answer, reference_answer, retrieved_docs, relevant_docs, question)
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1_score = metrics["f1"]
        f2_score = metrics["f2"]
        
        # 记录结果
        single_result = {
            "test_case_id": idx + 1,
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f2_score": f2_score,
            "answer_length": len(model_answer),
            "reference_length": len(reference_answer),
            "retrieved_count": len(retrieved_docs),
            "relevant_count": len(relevant_docs),
            "question_type": question_type,
            "test_mode": test_mode,
            "model_name": "internlm2_5-7b-chat",
            "generation_time": generation_time,
            "retrieval_time": retrieval_time,
            "total_time": total_time,
            "generation_success": model_answer != "未知" and "生成失败" not in model_answer
        }
        
        # 输出结果
        print(f"模型回答: {model_answer}")
        print(f"标准答案: {reference_answer[:80]}..." if len(reference_answer) > 80 else f"标准答案: {reference_answer}")
        print(f"准确率: {precision:.4f} | 召回率: {recall:.4f} | F1: {f1_score:.4f} | F2: {f2_score:.4f}")
        print(f"时间: 检索{retrieval_time:.2f}s + 生成{generation_time:.2f}s = 总计{total_time:.2f}s")
        
        if recall == 1 and precision == 0:
            print("⚠️  检索成功但回答错误!")
            # 调试：显示评估详情
            print(f"\n🔍 评估详情:")
            print(f"  相似度: {calculate_text_similarity(model_answer, reference_answer):.4f}")
            
            if question_type == "basic_info":
                model_company = extract_company_name(model_answer)
                ref_company = extract_company_name(reference_answer)
                print(f"  公司名匹配: 模型'{model_company}' vs 标准'{ref_company}'")
                
                model_product = extract_product_name(model_answer)
                ref_product = extract_product_name(reference_answer)
                print(f"  产品名匹配: 模型'{model_product}' vs 标准'{ref_product}'")
        
        # 记录到TensorBoard
        log_to_tensorboard(tb_writer, step=idx+1, metrics={
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f2_score": f2_score,
            "answer_length": len(model_answer),
            "reference_length": len(reference_answer),
            "retrieved_count": len(retrieved_docs),
            "generation_time": generation_time,
            "retrieval_time": retrieval_time,
            "total_time": total_time
        })
        
        test_results.append(single_result)
        
        # 每10条输出一次进度
        if (idx + 1) % 10 == 0:
            avg_precision = sum([r["precision"] for r in test_results]) / len(test_results)
            avg_recall = sum([r["recall"] for r in test_results]) / len(test_results)
            avg_f1 = sum([r["f1_score"] for r in test_results]) / len(test_results)
            avg_f2 = sum([r["f2_score"] for r in test_results]) / len(test_results)
            avg_generation_time = sum([r["generation_time"] for r in test_results]) / len(test_results)
            success_rate = successful_generations / (successful_generations + failed_generations) * 100 if (successful_generations + failed_generations) > 0 else 0
            
            print(f"\n📊 进度: {idx+1}/{len(test_cases)}，平均准确率: {avg_precision:.4f}，平均召回率: {avg_recall:.4f}")
            print(f"平均F1: {avg_f1:.4f}，平均F2: {avg_f2:.4f}")
            print(f"生成成功率: {success_rate:.1f}% ({successful_generations}成功/{failed_generations}失败)")
            
            # 将平均指标也记录到TensorBoard
            tb_writer.add_scalar("平均指标/准确率", avg_precision, idx+1)
            tb_writer.add_scalar("平均指标/召回率", avg_recall, idx+1)
            tb_writer.add_scalar("平均指标/F1分数", avg_f1, idx+1)
            tb_writer.add_scalar("平均指标/F2分数", avg_f2, idx+1)
            tb_writer.add_scalar("平均指标/生成时间", avg_generation_time, idx+1)
            tb_writer.add_scalar("平均指标/生成成功率", success_rate, idx+1)
        
        # 每50条保存一次中间结果
        if (idx + 1) % 50 == 0:
            temp_result_file = os.path.join(result_dir, f"temp_results_{timestamp}_{idx+1}.json")
            with open(temp_result_file, "w", encoding="utf-8") as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"💾 已保存中间结果到: {temp_result_file}")
    
    # 6. 保存测试结果
    if test_mode == "with_prompt":
        result_file_name = f"InternLM2_5_With_Prompt_Test_{timestamp}.json"
    else:
        result_file_name = f"InternLM2_5_Without_Prompt_Test_{timestamp}.json"
    
    result_file_path = os.path.join(result_dir, result_file_name)
    
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试结果已保存: {result_file_path}")
    
    # 7. 生成详细统计报告（包含F1和F2指标）
    generate_detailed_statistics(test_results, output_dir, test_mode, tb_writer, successful_generations, failed_generations)
    
    # 8. 输出总结
    print("\n" + "="*60)
    print("🎉 测试完成!")
    print("="*60)
    
    # 关闭TensorBoard
    tb_writer.close()
    
    return test_results

def generate_detailed_statistics(test_results, output_dir, test_mode, tb_writer=None, successful_generations=0, failed_generations=0):
    """
    生成详细统计报告，包含F1和F2指标
    """
    if not test_results:
        print("❌ 无测试结果，无法生成统计")
        return
    
    if test_mode == "with_prompt":
        stats_file = os.path.join(output_dir, "detailed_statistics_with_prompt.txt")
    else:
        stats_file = os.path.join(output_dir, "detailed_statistics_without_prompt.txt")
    
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        if test_mode == "with_prompt":
            f.write("InternLM2.5模型测试详细统计（使用提示词模板）\n")
        else:
            f.write("InternLM2.5模型测试详细统计（不使用提示词模板）\n")
        f.write("="*60 + "\n\n")
        
        # 总体统计
        total_cases = len(test_results)
        avg_precision = sum([r.get("precision", 0) for r in test_results]) / total_cases
        avg_recall = sum([r.get("recall", 0) for r in test_results]) / total_cases
        avg_f1 = sum([r.get("f1_score", 0) for r in test_results]) / total_cases
        avg_f2 = sum([r.get("f2_score", 0) for r in test_results]) / total_cases
        avg_retrieved = sum([r["retrieved_count"] for r in test_results]) / total_cases
        avg_generation_time = sum([r["generation_time"] for r in test_results]) / total_cases
        avg_total_time = sum([r["total_time"] for r in test_results]) / total_cases
        
        f.write("📊 总体统计（包含F1和F2指标）:\n")
        f.write(f"   总测试用例: {total_cases}\n")
        f.write(f"   平均准确率(Precision): {avg_precision:.4f}\n")
        f.write(f"   平均召回率(Recall): {avg_recall:.4f}\n")
        f.write(f"   平均F1分数: {avg_f1:.4f}\n")
        f.write(f"   平均F2分数: {avg_f2:.4f}\n")
        f.write(f"   平均检索文档数: {avg_retrieved:.1f}\n")
        f.write(f"   平均生成时间: {avg_generation_time:.2f}s\n")
        f.write(f"   平均总时间: {avg_total_time:.2f}s\n")
        f.write(f"   生成成功率: {successful_generations}/{total_cases} ({successful_generations/total_cases*100:.1f}%)\n\n")
        
        # 记录到TensorBoard
        if tb_writer:
            tb_writer.add_scalar("最终统计/平均准确率", avg_precision, 0)
            tb_writer.add_scalar("最终统计/平均召回率", avg_recall, 0)
            tb_writer.add_scalar("最终统计/平均F1分数", avg_f1, 0)
            tb_writer.add_scalar("最终统计/平均F2分数", avg_f2, 0)
            tb_writer.add_scalar("最终统计/平均生成时间", avg_generation_time, 0)
            tb_writer.add_scalar("最终统计/生成成功率", successful_generations/total_cases*100, 0)
        
        # 计算micro和macro平均
        f.write(f"📈 综合指标:\n")
        
        # 计算micro F1: 使用平均precision和recall计算
        if avg_precision + avg_recall > 0:
            f1_micro = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            f2_micro = 5 * avg_precision * avg_recall / (4 * avg_precision + avg_recall)
            f.write(f"   Micro F1: {f1_micro:.4f}\n")
            f.write(f"   Micro F2: {f2_micro:.4f}\n")
            f.write(f"   Macro F1: {avg_f1:.4f}\n")
            f.write(f"   Macro F2: {avg_f2:.4f}\n")
        
        f.write("\n")
        
        # 准确率分布
        precision_dist = defaultdict(int)
        for res in test_results:
            precision_score = res.get("precision", 0)
            precision_dist[precision_score] += 1
        
        f.write("📈 准确率分布:\n")
        for score in sorted(precision_dist.keys()):
            count = precision_dist[score]
            percentage = count / total_cases * 100
            f.write(f"   准确率 {score}: {count} 条 ({percentage:.1f}%)\n")
        
        f.write("\n")
        
        # 召回率分布
        recall_dist = defaultdict(int)
        for res in test_results:
            recall_score = res.get("recall", 0)
            recall_dist[recall_score] += 1
        
        f.write("📈 召回率分布:\n")
        for score in sorted(recall_dist.keys()):
            count = recall_dist[score]
            percentage = count / total_cases * 100
            f.write(f"   召回率 {score}: {count} 条 ({percentage:.1f}%)\n")
        
        f.write("\n")
        
        # F1分数分布
        f1_dist = defaultdict(int)
        for res in test_results:
            f1_score = res.get("f1_score", 0)
            # 将F1分数四舍五入到小数点后1位进行分组
            f1_group = round(f1_score, 1)
            f1_dist[f1_group] += 1
        
        f.write("📈 F1分数分布:\n")
        for score in sorted(f1_dist.keys()):
            count = f1_dist[score]
            percentage = count / total_cases * 100
            f.write(f"   F1 {score:.1f}: {count} 条 ({percentage:.1f}%)\n")
        
        f.write("\n")
        
        # 按问题类型统计
        type_stats = defaultdict(lambda: {
            "total": 0, 
            "precision_sum": 0, 
            "recall_sum": 0, 
            "f1_sum": 0,
            "f2_sum": 0,
            "gen_time": []
        })
        
        for res in test_results:
            q_type = res["question_type"]
            type_stats[q_type]["total"] += 1
            type_stats[q_type]["precision_sum"] += res.get("precision", 0)
            type_stats[q_type]["recall_sum"] += res.get("recall", 0)
            type_stats[q_type]["f1_sum"] += res.get("f1_score", 0)
            type_stats[q_type]["f2_sum"] += res.get("f2_score", 0)
            type_stats[q_type]["gen_time"].append(res["generation_time"])
        
        f.write("🔍 按问题类型统计（包含F1和F2）:\n")
        for q_type, stats in sorted(type_stats.items()):
            total = stats["total"]
            precision_avg = stats["precision_sum"] / total if total > 0 else 0
            recall_avg = stats["recall_sum"] / total if total > 0 else 0
            f1_avg = stats["f1_sum"] / total if total > 0 else 0
            f2_avg = stats["f2_sum"] / total if total > 0 else 0
            avg_gen_time = sum(stats["gen_time"]) / total if total > 0 else 0
            
            f.write(f"\n   {q_type}:\n")
            f.write(f"       用例数: {total}\n")
            f.write(f"       准确率: {precision_avg:.4f}\n")
            f.write(f"       召回率: {recall_avg:.4f}\n")
            f.write(f"       F1分数: {f1_avg:.4f}\n")
            f.write(f"       F2分数: {f2_avg:.4f}\n")
            f.write(f"       平均生成时间: {avg_gen_time:.2f}s\n")
        
        f.write("\n")
        
        # 错误案例分析
        f.write("⚠️  错误案例分析:\n")
        error_cases = [r for r in test_results if r.get("precision", 0) == 0]
        
        if error_cases:
            # 按问题类型分组
            error_by_type = defaultdict(list)
            for case in error_cases:
                error_by_type[case["question_type"]].append(case)
            
            for q_type, cases in error_by_type.items():
                f.write(f"\n   {q_type} 错误 ({len(cases)} 条):\n")
                for case in cases[:3]:  # 每个类型显示前3条
                    f.write(f"       - {case['question'][:50]}...\n")
                    f.write(f"         模型: {case['model_answer'][:50]}...\n")
                    f.write(f"         标准: {case['reference_answer'][:50]}...\n")
                    f.write(f"         召回率: {case['recall']:.4f}, F1: {case['f1_score']:.4f}\n")
        else:
            f.write("   无错误案例！\n")
        
        # 性能分析
        f.write("\n" + "="*60 + "\n")
        f.write("📈 性能分析:\n")
        generation_times = [r["generation_time"] for r in test_results]
        if generation_times:
            f.write(f"   生成时间统计:\n")
            f.write(f"       最短: {min(generation_times):.2f}s\n")
            f.write(f"       最长: {max(generation_times):.2f}s\n")
            f.write(f"       中位数: {np.median(generation_times):.2f}s\n")
            f.write(f"       标准差: {np.std(generation_times):.2f}s\n")
        
        # 生成失败分析
        if failed_generations > 0:
            f.write(f"\n⚠️  生成失败分析:\n")
            f.write(f"   生成失败次数: {failed_generations}\n")
            f.write(f"   生成成功率: {successful_generations/total_cases*100:.1f}%\n")
            f.write(f"   主要失败类型: 张量维度不匹配错误\n")
    
    print(f"📊 详细统计已保存: {stats_file}")

# ====================== 14. 对比分析函数 ======================
def compare_two_modes(results_without_prompt, results_with_prompt):
    """
    对比两种模式的结果
    """
    print("\n" + "="*80)
    print("两种调研模式对比分析")
    print("="*80)
    
    if not results_without_prompt or not results_with_prompt:
        print("❌ 无法对比，至少需要两种模式的结果")
        return
    
    # 创建对比目录
    compare_dir = "./output_compare"
    os.makedirs(compare_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    compare_file = os.path.join(compare_dir, f"对比报告_{timestamp}.txt")
    
    # 计算总体指标
    total_cases = len(results_without_prompt)
    
    # 计算平均指标
    avg_precision_without = sum([r.get("precision", 0) for r in results_without_prompt]) / total_cases
    avg_recall_without = sum([r.get("recall", 0) for r in results_without_prompt]) / total_cases
    avg_f1_without = sum([r.get("f1_score", 0) for r in results_without_prompt]) / total_cases
    avg_f2_without = sum([r.get("f2_score", 0) for r in results_without_prompt]) / total_cases
    avg_gen_time_without = sum([r["generation_time"] for r in results_without_prompt]) / total_cases
    
    avg_precision_with = sum([r.get("precision", 0) for r in results_with_prompt]) / total_cases
    avg_recall_with = sum([r.get("recall", 0) for r in results_with_prompt]) / total_cases
    avg_f1_with = sum([r.get("f1_score", 0) for r in results_with_prompt]) / total_cases
    avg_f2_with = sum([r.get("f2_score", 0) for r in results_with_prompt]) / total_cases
    avg_gen_time_with = sum([r["generation_time"] for r in results_with_prompt]) / total_cases
    
    # 计算提升
    precision_improvement = avg_precision_with - avg_precision_without
    recall_improvement = avg_recall_with - avg_recall_without
    f1_improvement = avg_f1_with - avg_f1_without
    f2_improvement = avg_f2_with - avg_f2_without
    gen_time_change = avg_gen_time_with - avg_gen_time_without
    
    with open(compare_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("两种调研模式对比报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("调研目标:\n")
        f.write("1. 第一种调研（不使用提示词）：不做任何处理情况下，观察模型的回答情况\n")
        f.write("2. 第二种调研（使用提示词）：构建提示词模板对大模型加以训练\n\n")
        
        f.write("📊 总体对比:\n")
        f.write(f"                 | 不使用提示词 | 使用提示词 | 变化\n")
        f.write(f"-----------------|------------|------------|------------\n")
        f.write(f"测试用例数       | {total_cases:^10} | {total_cases:^10} | -\n")
        f.write(f"平均准确率       | {avg_precision_without:.4f}     | {avg_precision_with:.4f}     | {precision_improvement:+.4f}\n")
        f.write(f"平均召回率       | {avg_recall_without:.4f}     | {avg_recall_with:.4f}     | {recall_improvement:+.4f}\n")
        f.write(f"平均F1分数       | {avg_f1_without:.4f}     | {avg_f1_with:.4f}     | {f1_improvement:+.4f}\n")
        f.write(f"平均F2分数       | {avg_f2_without:.4f}     | {avg_f2_with:.4f}     | {f2_improvement:+.4f}\n")
        f.write(f"平均生成时间     | {avg_gen_time_without:.2f}s   | {avg_gen_time_with:.2f}s   | {gen_time_change:+.2f}s\n\n")
        
        # 按问题类型对比
        f.write("🔍 按问题类型对比:\n")
        
        # 按类型分组
        type_results_without = defaultdict(list)
        type_results_with = defaultdict(list)
        
        for result in results_without_prompt:
            type_results_without[result["question_type"]].append(result)
        
        for result in results_with_prompt:
            type_results_with[result["question_type"]].append(result)
        
        for q_type in sorted(set(list(type_results_without.keys()) + list(type_results_with.keys()))):
            if q_type in type_results_without and q_type in type_results_with:
                without_list = type_results_without[q_type]
                with_list = type_results_with[q_type]
                
                if without_list and with_list:
                    precision_without_avg = sum([r.get("precision", 0) for r in without_list]) / len(without_list)
                    precision_with_avg = sum([r.get("precision", 0) for r in with_list]) / len(with_list)
                    f1_without_avg = sum([r.get("f1_score", 0) for r in without_list]) / len(without_list)
                    f1_with_avg = sum([r.get("f1_score", 0) for r in with_list]) / len(with_list)
                    gen_time_without_avg = sum([r["generation_time"] for r in without_list]) / len(without_list)
                    gen_time_with_avg = sum([r["generation_time"] for r in with_list]) / len(with_list)
                    
                    precision_improvement = precision_with_avg - precision_without_avg
                    f1_improvement = f1_with_avg - f1_without_avg
                    time_change = gen_time_with_avg - gen_time_without_avg
                    precision_sign = "↑" if precision_improvement > 0 else "↓" if precision_improvement < 0 else "→"
                    f1_sign = "↑" if f1_improvement > 0 else "↓" if f1_improvement < 0 else "→"
                    time_sign = "↑" if time_change > 0 else "↓" if time_change < 0 else "→"
                    
                    f.write(f"\n{q_type:25}:\n")
                    f.write(f"    不使用提示词: 准确率{precision_without_avg:.4f}, F1{f1_without_avg:.4f}, 生成时间{gen_time_without_avg:.2f}s\n")
                    f.write(f"    使用提示词: 准确率{precision_with_avg:.4f}, F1{f1_with_avg:.4f}, 生成时间{gen_time_with_avg:.2f}s\n")
                    f.write(f"    准确率变化: {precision_improvement:+.4f} {precision_sign}\n")
                    f.write(f"    F1变化: {f1_improvement:+.4f} {f1_sign}\n")
                    f.write(f"    生成时间变化: {time_change:+.2f}s {time_sign}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("结论:\n")
        
        if f1_improvement > 0.1:
            f.write(f"✅ 提示词模板显著提高了模型表现，F1分数提升了 {f1_improvement*100:.2f}%\n")
        elif f1_improvement > 0:
            f.write(f"✅ 提示词模板在一定程度上提高了模型表现，F1分数提升了 {f1_improvement*100:.2f}%\n")
        elif f1_improvement == 0:
            f.write("⚠️  提示词模板对模型表现没有明显影响\n")
        else:
            f.write(f"❌ 提示词模板反而降低了模型表现，F1分数降低了 {-f1_improvement*100:.2f}%\n")
        
        f.write("\n建议:\n")
        if f1_improvement > 0:
            f.write("1. 提示词模板在招投标领域是有效的\n")
            f.write("2. 可以考虑进一步优化提示词模板以获得更好的效果\n")
            if gen_time_change > 0:
                f.write(f"3. 提示词模板增加了生成时间({gen_time_change:.2f}s)，可能需要优化提示词长度\n")
            else:
                f.write(f"3. 提示词模板减少了生成时间({-gen_time_change:.2f}s)，这是一个积极的信号\n")
        else:
            f.write("1. 可能需要重新设计提示词模板\n")
            f.write("2. 可以考虑使用更细致的指令或示例\n")
            f.write("3. 检查模型的fine-tuning情况\n")
    
    # 控制台输出
    print(f"\n📊 总体对比:")
    print(f"                 | 不使用提示词 | 使用提示词 | 变化")
    print(f"-----------------|------------|------------|------------")
    print(f"测试用例数       | {total_cases:^10} | {total_cases:^10} | -")
    print(f"平均准确率       | {avg_precision_without:.4f}     | {avg_precision_with:.4f}     | {precision_improvement:+.4f}")
    print(f"平均召回率       | {avg_recall_without:.4f}     | {avg_recall_with:.4f}     | {recall_improvement:+.4f}")
    print(f"平均F1分数       | {avg_f1_without:.4f}     | {avg_f1_with:.4f}     | {f1_improvement:+.4f}")
    print(f"平均F2分数       | {avg_f2_without:.4f}     | {avg_f2_with:.4f}     | {f2_improvement:+.4f}")
    print(f"平均生成时间     | {avg_gen_time_without:.2f}s   | {avg_gen_time_with:.2f}s   | {gen_time_change:+.2f}s")
    
    print(f"\n🎯 提示词模板效果总结:")
    if f1_improvement > 0.1:
        print(f"  ✅ 提示词模板显著有效，F1分数提升了 {f1_improvement*100:.2f}%")
    elif f1_improvement > 0:
        print(f"  ✅ 提示词模板有一定效果，F1分数提升了 {f1_improvement*100:.2f}%")
    elif f1_improvement == 0:
        print(f"  ⚠️  提示词模板效果不明显")
    else:
        print(f"  ❌ 提示词模板有负作用，F1分数降低了 {-f1_improvement*100:.2f}%")
    
    print(f"\n⏱️  性能影响:")
    if gen_time_change > 0:
        print(f"  ⚠️  提示词模板增加了生成时间: {gen_time_change:.2f}s")
    elif gen_time_change < 0:
        print(f"  ✅ 提示词模板减少了生成时间: {-gen_time_change:.2f}s")
    else:
        print(f"  ⚠️  提示词模板对生成时间无影响")
    
    print(f"\n📄 对比报告已保存: {compare_file}")

# ====================== 15. 主函数：支持多种运行模式 ======================
if __name__ == "__main__":
    """
    主函数：支持三种运行模式
    1. 只运行第一种调研（不使用提示词）
    2. 只运行第二种调研（使用提示词）
    3. 同时运行两种调研并进行对比
    """
    print("="*80)
    print("InternLM2.5 模型调研测试")
    print("="*80)
    print("\n请选择运行模式：")
    print("1. 第一种调研：不使用提示词模板（观察原始模型表现）")
    print("2. 第二种调研：使用提示词模板（测试提示词效果）")
    print("3. 同时运行两种调研并进行对比分析")
    print("0. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (0-3): ").strip()
            
            if choice == "0":
                print("程序退出")
                break
            elif choice == "1":
                # 只运行第一种调研
                print("\n" + "="*80)
                print("开始第一种调研：不使用提示词模板")
                print("="*80)
                
                start_time = time.time()
                results_without_prompt = run_optimized_internlm2_5_test(test_mode="without_prompt")
                end_time = time.time()
                
                print(f"\n⏱️  第一种调研完成，耗时: {end_time - start_time:.2f} 秒")
                
                # 输出TensorBoard使用说明
                print("\n📊 TensorBoard使用说明:")
                print("1. 打开新的终端窗口")
                print("2. 运行命令: tensorboard --logdir=./output_internlm2.5_without_prompt/tb_logs_*")
                print("3. 在浏览器中访问: http://localhost:6006")
                break
                
            elif choice == "2":
                # 只运行第二种调研
                print("\n" + "="*80)
                print("开始第二种调研：使用提示词模板")
                print("="*80)
                
                start_time = time.time()
                results_with_prompt = run_optimized_internlm2_5_test(test_mode="with_prompt")
                end_time = time.time()
                
                print(f"\n⏱️  第二种调研完成，耗时: {end_time - start_time:.2f} 秒")
                
                # 输出TensorBoard使用说明
                print("\n📊 TensorBoard使用说明:")
                print("1. 打开新的终端窗口")
                print("2. 运行命令: tensorboard --logdir=./output_internlm2.5_with_prompt/tb_logs_*")
                print("3. 在浏览器中访问: http://localhost:6006")
                break
                
            elif choice == "3":
                # 同时运行两种调研
                print("\n" + "="*80)
                print("开始同时运行两种调研模式")
                print("="*80)
                
                # 运行第一种调研
                print("\n>>> 第一阶段：运行第一种调研（不使用提示词）")
                start_time1 = time.time()
                results_without_prompt = run_optimized_internlm2_5_test(test_mode="without_prompt")
                end_time1 = time.time()
                
                print(f"\n⏱️  第一种调研完成，耗时: {end_time1 - start_time1:.2f} 秒")
                
                # 等待2秒，让资源释放
                print("\n等待2秒，准备开始第二种调研...")
                time.sleep(2)
                
                # 运行第二种调研
                print("\n>>> 第二阶段：运行第二种调研（使用提示词）")
                start_time2 = time.time()
                results_with_prompt = run_optimized_internlm2_5_test(test_mode="with_prompt")
                end_time2 = time.time()
                
                print(f"\n⏱️  第二种调研完成，耗时: {end_time2 - start_time2:.2f} 秒")
                
                # 对比分析
                print("\n>>> 第三阶段：对比分析两种模式")
                compare_two_modes(results_without_prompt, results_with_prompt)
                
                total_time = (end_time1 - start_time1) + (end_time2 - start_time2)
                print(f"\n⏱️  总耗时: {total_time:.2f} 秒")
                
                # 输出TensorBoard使用说明
                print("\n📊 TensorBoard使用说明:")
                print("1. 第一种调研TensorBoard:")
                print("   tensorboard --logdir=./output_internlm2.5_without_prompt/tb_logs_*")
                print("2. 第二种调研TensorBoard:")
                print("   tensorboard --logdir=./output_internlm2.5_with_prompt/tb_logs_*")
                print("3. 在浏览器中访问: http://localhost:6006")
                break
                
            else:
                print("❌ 无效选择，请输入0-3之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*80)
    print("程序运行结束")
    print("="*80)