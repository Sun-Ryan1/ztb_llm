import torch
import json
import numpy as np
import faiss
import os
import re
import sys
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from torch.utils.tensorboard import SummaryWriter
from difflib import SequenceMatcher

# 全局问题类型统计
question_type_stats = {
    "company_legal_representative": {"total": 0, "correct": 0, "recall": 0},
    "price_info": {"total": 0, "correct": 0, "recall": 0},
    "supplier_info": {"total": 0, "correct": 0, "recall": 0},
    "buyer_info": {"total": 0, "correct": 0, "recall": 0},
    "bid_winner_info": {"total": 0, "correct": 0, "recall": 0},
    "project_info": {"total": 0, "correct": 0, "recall": 0},
    "company_basic_info": {"total": 0, "correct": 0, "recall": 0},
    "general_info": {"total": 0, "correct": 0, "recall": 0}
}

# ====================== 1. 智能提示词选择器 ======================
class SmartPromptSelector:
    """智能提示词选择器
    根据问题类型、内容、检索结果动态选择最合适的提示词策略
"""
    
    def __init__(self):
        # 不同问题类型的最佳提示词策略配置
        self.prompt_strategies = {
            # 强招投标相关类型
"""分析问题复杂度
"""question_lower = question.lower()
        
        # 判断是否招投标相关问题
        bidding_score = sum(1 for kw in self.bidding_keywords if kw in question_lower)
        general_score = sum(1 for kw in self.general_keywords if kw in question_lower)
        
        # 判断问题长度和复杂度
        length_score = len(question) / 100  # 归一化
        has_special_chars = bool(re.search(r'[、，。；：？]', question))
        
        # 综合评分
        complexity_score = (
            0.4 * min(bidding_score / 3, 1.0) +  # 招投标相关性
            0.3 * min(general_score / 5, 1.0) +  # 通用性
            0.2 * min(length_score, 1.0) +       # 长度
            0.1 * (1 if has_special_chars else 0)  # 特殊字符
        )
        
        return {
            "is_bidding_related": bidding_score > 0,
            "bidding_score": bidding_score,
            "complexity": complexity_score,
            "length": len(question),
            "has_special_chars": has_special_chars
        }
    
    def analyze_retrieval_confidence(self, retrieved_docs, question):
        """分析检索结果的置信度
"""
        if not retrieved_docs:
            return {"confidence": 0.0, "relevant_count": 0, "avg_similarity": 0.0}
        
        # 提取问题中的关键实体
        question_entities = self.extract_entities(question)
        
        relevant_count = 0
        total_similarity = 0
        
        for doc in retrieved_docs:
            # 检查文档是否包含问题实体
            doc_entities = self.extract_entities(doc)
            common_entities = set(question_entities) & set(doc_entities)
            
            # 计算文本相似度
            similarity = SequenceMatcher(None, question, doc).ratio()
            
            if common_entities or similarity > 0.3:
                relevant_count += 1
                total_similarity += similarity
        
        avg_similarity = total_similarity / len(retrieved_docs) if retrieved_docs else 0
        confidence = min(relevant_count / len(retrieved_docs), 1.0) * 0.7 + avg_similarity * 0.3
        
        return {
            "confidence": confidence,
            "relevant_count": relevant_count,
            "avg_similarity": avg_similarity,
            "total_docs": len(retrieved_docs)
        }
    
    def extract_entities(self, text):
        """提取文本中的实体
"""
        entities = []
        
        # 公司名称
        company_patterns = [
            r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团|分公司|股份公司|有限责任公司)',
            r'([\u4e00-\u9fa5]{2,6})(?:公司|厂|店|中心|局|所|院)'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            entities.extend([m for m in matches if isinstance(m, str)])
        
        # 人名
        name_patterns = [
            r'法定代表人是([\u4e00-\u9fa5]{2,4})',
            r'法人代表是([\u4e00-\u9fa5]{2,4})',
            r'负责人是([\u4e00-\u9fa5]{2,4})'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            entities.extend([m for m in matches if isinstance(m, str)])
        
        # 价格
        price_matches = re.findall(r'(\d+\.?\d*)[万万千]?元', text)
        entities.extend([f"{price}元" for price in price_matches])
        
        # 项目名称
        project_matches = re.findall(r'项目名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})', text)
        entities.extend(project_matches)
        
        return list(set(entities))
    
    def select_prompt_strategy(self, question, question_type, retrieved_docs):
        """智能选择提示词策略
        返回: (strategy_type, confidence, reason)
"""
        # 分析问题和检索结果
        question_analysis = self.analyze_question_complexity(question)
        retrieval_analysis = self.analyze_retrieval_confidence(retrieved_docs, question)
        
        # 获取该问题类型的默认策略
        default_strategy = self.prompt_strategies.get(question_type, {
            "strategy": "adaptive",
            "confidence_threshold": 0.5,
            "fallback": "simple"
        })
        
        # 核心问题类型直接使用专业策略，提高准确率
        core_question_types = ["company_legal_representative", "price_info", "supplier_info", 
                             "bid_winner_info", "buyer_info", "project_info"]
        
        # 动态调整策略
        retrieval_confidence = retrieval_analysis["confidence"]
        is_bidding_related = question_analysis["is_bidding_related"]
        question_complexity = question_analysis["complexity"]
        
        reason_parts = []
        
        # 对于核心问题类型，强制使用专业策略
        if question_type in core_question_types:
            strategy_type = "professional"
            reason_parts.append("核心问题类型，使用专业策略")
        else:
            # 获取策略类型
            strategy_type = default_strategy["strategy"]
            confidence_threshold = max(default_strategy["confidence_threshold"] * 0.4, 0.2)  # 进一步降低阈值
            
            if strategy_type == "professional":
                # 专业策略：大幅降低置信度要求
                if retrieval_confidence < confidence_threshold:
                    strategy_type = default_strategy["fallback"]
                    reason_parts.append(f"检索置信度({retrieval_confidence:.2f})低于阈值({confidence_threshold})")
                # 放宽招投标相关判断
                if not is_bidding_related:
                    strategy_type = "simple"
                    reason_parts.append("问题与招投标无关")
            
            elif strategy_type == "adaptive":
                # 自适应策略：进一步降低专业评分阈值
                professional_score = (
                    0.4 * (1 if is_bidding_related else 0) +
                    0.3 * min(question_analysis["bidding_score"] / 1.5, 1.0) +  # 进一步降低分母，提高评分
                    0.2 * retrieval_confidence +
                    0.1 * question_complexity
                )
                
                if professional_score > 0.3:  # 进一步降低阈值
                    strategy_type = "professional"
                    reason_parts.append(f"自适应评分高({professional_score:.2f})")
                else:
                    strategy_type = "simple"
                    reason_parts.append(f"自适应评分低({professional_score:.2f})")
        
        # 大幅放宽最终检查：只有当检索结果非常差时，才强制使用简单策略
        if retrieval_confidence < 0.05:  # 大幅降低阈值
            strategy_type = "simple"
            reason_parts.append("检索质量太差")
        
        reason = "，".join(reason_parts) if reason_parts else "使用默认策略"
        
        return {
            "strategy": strategy_type,
            "confidence": retrieval_confidence,
            "reason": reason,
            "question_analysis": question_analysis,
            "retrieval_analysis": retrieval_analysis
        }

# ====================== 2. 初始化output目录 ======================
def init_output_dir(test_mode="with_prompt"):
    """根据测试模式初始化输出目录
    test_mode: "with_prompt"
"""
    if test_mode == "with_prompt":
        output_dir = "./output_glm3_with_prompt"
    elif test_mode == "without_prompt":
        output_dir = "./output_glm3_without_prompt"
    else:  # smart_prompt
        output_dir = "./output_glm3_smart_prompt"
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_log_dir = os.path.join(output_dir, f"tb_logs_{timestamp}")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    return output_dir, tb_log_dir, result_dir, log_dir

# ====================== 3. 日志重定向 ======================
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

# ====================== 4. TensorBoard初始化 ======================
def init_tensorboard(tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"✅ TensorBoard日志已初始化，存储路径：{tb_log_dir}")
    return writer

def log_to_tensorboard(writer, step, metrics, test_mode):
    """记录到TensorBoard
    根据测试模式添加不同的前缀
"""
    prefix = f"{test_mode}/"
    
    writer.add_scalar(f"{prefix}召回率/单条用例", metrics["recall_score"], step)
    writer.add_scalar(f"{prefix}准确率/单条用例", metrics["accuracy"], step)
    writer.add_scalar(f"{prefix}回答长度/模型生成", metrics["answer_length"], step)
    writer.add_scalar(f"{prefix}回答长度/标准答案", metrics["reference_length"], step)
    writer.add_scalar(f"{prefix}检索文档数", metrics["retrieved_count"], step)
    
    # 记录提示词策略相关信息
    if "prompt_strategy" in metrics:
        # 安全地访问prompt_strategy字段
        strategy_info = metrics["prompt_strategy"]
        if "confidence" in strategy_info:
            writer.add_scalar(f"{prefix}提示词策略/置信度", strategy_info["confidence"], step)
        
        # 检查是否有retrieval_analysis字段
        if "retrieval_analysis" in strategy_info and "relevant_count" in strategy_info["retrieval_analysis"]:
            writer.add_scalar(f"{prefix}提示词策略/检索相关文档数", strategy_info["retrieval_analysis"]["relevant_count"], step)

# ====================== 5. 配置量化参数 ======================
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# ====================== 6. 加载GLM3模型 ======================
def load_glm3_model(glm3_path):
    """专门加载GLM3模型，适配GLM3的特殊设置
"""
    print(f"正在加载GLM3模型：{glm3_path}")
    
    try:
        # GLM3的tokenizer加载
        tokenizer = AutoTokenizer.from_pretrained(
            glm3_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True
        )
        
        # 对于ChatGLM3，pad_token需要特殊处理
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 检查tokenizer是否有特殊token
        print(f"✅ GLM3 Tokenizer加载成功")
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   eos_token: {tokenizer.eos_token}")
        print(f"   bos_token: {tokenizer.bos_token}")
        
        # 检查是否有apply_chat_template方法
        if hasattr(tokenizer, 'apply_chat_template'):
            print(f"   tokenizer支持apply_chat_template")
        else:
            print(f"   tokenizer不支持apply_chat_template，将使用手动构建对话格式")
        
        # GLM3模型加载
        llm_model = AutoModelForCausalLM.from_pretrained(
            glm3_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        llm_model.eval()
        print(f"✅ GLM3模型加载成功")
        
        return tokenizer, llm_model
        
    except Exception as e:
        print(f"❌ GLM3模型加载失败，错误信息：{e}")
        import traceback
        traceback.print_exc()
        return None, None

# ====================== 7. 加载BGE Embedding模型 ======================
def load_bge_embedding_model(embedding_path):
    """加载BGE Embedding模型
"""
    print(f"正在加载BGE Embedding模型：{embedding_path}")
    
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = AutoModel.from_pretrained(
            embedding_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True
        ).to(device)
        
        embedding_model.eval()
        print(f"✅ BGE Embedding模型加载成功，使用设备：{device}")
        
        return embedding_tokenizer, embedding_model, device
        
    except Exception as e:
        print(f"❌ BGE Embedding模型加载失败，错误信息：{e}")
        return None, None, None

# ====================== 8. BGE文本转向量函数 ======================
def bge_embedding_encode(embedding_models, text, batch_mode=False):
    """BGE文本编码函数
"""
    embedding_tokenizer, embedding_model, *_ = embedding_models
    
    if batch_mode:
        # 批量模式：text是文档列表，不转换为字符串
        if not text or not isinstance(text, list):
            return np.array([])
        # 过滤空文档
        valid_texts = [str(t).strip() for t in text if str(t).strip()]
        if not valid_texts:
            return np.array([])
        
        inputs = embedding_tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(embedding_model.device)
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        return embeddings.cpu().numpy()
    else:
        # 单条模式：正常处理
        text = str(text).strip()
        if not text:
            return np.array([])
        
        inputs = embedding_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(embedding_model.device)
        
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            hidden_state = outputs.last_hidden_state[:, 0]
            vec = hidden_state.cpu().numpy().squeeze()
        
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

# ====================== 9. 分析问题类型 ======================
def analyze_question_type(question):
    """分析问题类型，返回对应的类型标签
"""
    question_lower = question.lower()
    
    # 法定代表人相关
"""提取文本中的所有关键实体
"""entities = []
    
    # 1. 公司名称（增强模式）
    company_patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团|分公司|股份公司|有限责任公司)',
        r'([\u4e00-\u9fa5]{2,6})(?:公司|厂|店|中心|局|所|院)',
        r'供应商[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'采购方[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
        r'中标方[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, str) and len(match) > 1:
                entities.append(match)
    
    # 2. 人名（法定代表人）
    name_patterns = [
        r'法定代表人是([\u4e00-\u9fa5]{2,4})',
        r'法人代表是([\u4e00-\u9fa5]{2,4})',
        r'负责人是([\u4e00-\u9fa5]{2,4})',
        r'姓名为([\u4e00-\u9fa5]{2,4})'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m for m in matches if isinstance(m, str) and len(m) >= 2])
    
    # 3. 价格信息
    price_patterns = [
        r'(\d+\.?\d*)[万万千]?元',
        r'价格[为是](\d+\.?\d*)[万万千]?元',
        r'金额[为是](\d+\.?\d*)[万万千]?元',
        r'成交价[为是](\d+\.?\d*)[万万千]?元'
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, str) and match.replace('.', '').isdigit():
                entities.append(f"{match}元")
    
    # 4. 项目名称
    project_patterns = [
        r'项目名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})',
        r'([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})项目'
    ]
    
    for pattern in project_patterns:
        matches = re.findall(pattern, text)
        entities.extend([m for m in matches if isinstance(m, str) and len(m) > 3])
    
    return list(set(entities))

# ====================== 11. 增强的检索函数 ======================
def enhanced_retrieval_with_reranking(embedding_models, index, docs, question, top_k=10, similarity_threshold=0.03):
    """增强版检索，包含重新排序和问题类型优先级调整
"""
    results = []
    
    # 1. 向量检索（仅当索引有效时）
    if index is not None:
        question_vector = bge_embedding_encode(embedding_models, question)
        if question_vector.size > 0:
            try:
                # 恢复到原始检索数量
                distances, indices = index.search(question_vector.reshape(1, -1), top_k * 3)
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(docs) and dist > similarity_threshold:
                        results.append({
                            "doc": docs[idx],
                            "similarity": float(dist),
                            "type": "vector",
                            "index": idx
                        })
            except Exception as e:
                print(f"⚠️  向量检索失败，仅使用关键词检索: {e}")
    
    # 2. 关键词加强检索
    keyword_results = keyword_enhanced_retrieval(question, docs)
    results.extend(keyword_results)
    
    # 3. 基于问题类型的优先级调整
    question_type = analyze_question_type(question)
    
    for result in results:
        doc_text = result["doc"].lower()
        
        # 根据问题类型大幅调整相似度分数
        if question_type == "company_legal_representative":
            # 法定代表人问题权重增加
            if "法定代表人是" in doc_text or "法人代表是" in doc_text or "法定代表人：" in doc_text:
                result["similarity"] += 0.9
            elif any(keyword in doc_text for keyword in ["法定代表人", "法人", "负责人"]):
                result["similarity"] += 0.6
        
        elif question_type == "price_info":
            if "价格为" in doc_text or "价格是" in doc_text or "金额为" in doc_text or "金额：" in doc_text:
                result["similarity"] += 0.8
            elif any(keyword in doc_text for keyword in ["价格", "金额", "元", "费用", "成本"]):
                result["similarity"] += 0.4
        
        elif question_type == "supplier_info":
            if "供应商是" in doc_text or "供应商为" in doc_text or "供应商：" in doc_text:
                result["similarity"] += 0.8
            elif any(keyword in doc_text for keyword in ["供应商", "提供", "供货", "供应"]):
                result["similarity"] += 0.4
        
        elif question_type == "bid_winner_info":
            if "中标供应商为" in doc_text or "中标方是" in doc_text or "中标单位" in doc_text:
                result["similarity"] += 0.8
            elif any(keyword in doc_text for keyword in ["中标", "中标方", "中标供应商"]):
                result["similarity"] += 0.4
        
        elif question_type == "buyer_info":
            if "采购方为" in doc_text or "采购方是" in doc_text or "采购方：" in doc_text:
                result["similarity"] += 0.8
            elif any(keyword in doc_text for keyword in ["采购方", "购买方", "采购", "买方"]):
                result["similarity"] += 0.4
        
        elif question_type == "project_info":
            if "项目名称" in doc_text or "项目是" in doc_text or "项目：" in doc_text:
                result["similarity"] += 0.6
        
        # 实体匹配加分
        entities = extract_all_entities(question)
        doc_entities = extract_all_entities(doc_text)
        common_entities = set(entities) & set(doc_entities)
        if common_entities:
            result["similarity"] += 0.15 * len(common_entities)
    
    # 4. 去重和重新排序
    unique_results = []
    seen_docs = set()
    
    for result in results:
        doc_content = result["doc"]
        if doc_content not in seen_docs:
            seen_docs.add(doc_content)
            unique_results.append(result)
    
    unique_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 最终选择
    final_docs = []
    for result in unique_results[:top_k]:
        if result["similarity"] >= similarity_threshold:
            final_docs.append(result["doc"])
    
    return final_docs

def keyword_enhanced_retrieval(query, docs):
    """增强版关键词检索
"""
    results = []
    
    # 提取关键实体
    entities = extract_all_entities(query)
    
    for idx, doc in enumerate(docs):
        score = 0
        doc_lower = doc.lower()
        query_lower = query.lower()
        
        # 完全匹配实体
        for entity in entities:
            if entity in doc_lower:
                score += 1.0  # 增加完全匹配的分数
        
        # 部分匹配（子字符串）
        for entity in entities:
            if len(entity) > 2:
                # 检查是否是子字符串
                if any(entity in word for word in doc_lower.split()):
                    score += 0.5
        
        # 直接关键词匹配
        keywords = extract_keywords(query)
        for keyword in keywords:
            if keyword in doc_lower:
                score += 0.8
        
        # 问题类型关键词匹配
        question_type = analyze_question_type(query)
        if question_type == "company_legal_representative" and any(kw in doc_lower for kw in ["法定代表人", "法人代表"]):
            score += 0.5
        elif question_type == "price_info" and any(kw in doc_lower for kw in ["价格", "金额", "元"]):
            score += 0.5
        elif question_type == "supplier_info" and any(kw in doc_lower for kw in ["供应商", "提供"]):
            score += 0.5
        elif question_type == "bid_winner_info" and any(kw in doc_lower for kw in ["中标", "中标方"]):
            score += 0.5
        elif question_type == "buyer_info" and any(kw in doc_lower for kw in ["采购方", "购买方"]):
            score += 0.5
        
        if score > 0:
            results.append({
                "doc": doc,
                "similarity": min(score, 2.0),  # 上限提高
                "type": "keyword",
                "index": idx
            })
    
    return results

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
        "地址", "注册地址", "经营地址",
        "价格", "费用", "成本", "报价", "金额",
        "公司", "有限公司", "集团", "分公司"
    ]
    
    for keyword in bid_keywords:
        if keyword in text:
            keywords.append(keyword)
    
    return keywords

# ====================== 12. 智能提示词生成器 ======================
class SmartPromptGenerator:
    """智能提示词生成器
    根据策略生成不同风格的提示词
"""
    
    def __init__(self):
        # 专业提示词系统指令
        self.professional_system_prompt = """# 角色定位
你是聚焦招投标采购全流程的专业智能问答系统，需严格依据《招标投标法》《政府采购法》等法规，精准解答政策合规、业务操作、物资产品、电子系统操作等领域问题。

# 回答要求
1. 准确性：严格依据相关法规和政策，确保信息准确无误
2. 完整性：全面覆盖问题要点，提供详细的分析和解释
3. 专业性：正确使用专业术语，体现专业知识和分析能力
4. 清晰性：语言流畅，逻辑清晰，结构合理

# 任务说明
请根据提供的招投标相关信息，回答用户的提问。如果信息中有明确答案，请直接提取并回答。如果信息不完整或没有相关信息，请如实告知。"""
        
        # 简单提示词系统指令
        self.simple_system_prompt = """请根据提供的信息回答问题。如果信息中有明确答案，请直接回答。如果信息中没有相关信息，请如实告知。
"""
        
        # 自适应提示词系统指令
        self.adaptive_system_prompt = """请根据提供的信息，以清晰、准确的方式回答问题。注意使用适当的专业术语，但不要过度复杂化回答。
"""
    
    def get_professional_prompt(self, question_type, context, question):
        """获取专业提示词
"""
        # 根据不同问题类型定制具体指令
        if question_type == "company_legal_representative":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接回答法定代表人的姓名。如果信息中有明确的'法定代表人是XXX'，直接回答'XXX'。如果信息中没有相关信息，请回答'无法确定'。
"""
        
        elif question_type == "price_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接回答价格或金额信息。如果信息中有明确的'价格为XXX元'或'合同金额为XXX元'，直接回答'XXX元'。如果信息中没有价格信息，请如实告知。
"""
        
        elif question_type == "supplier_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接回答供应商信息。如果信息中有明确的'供应商是XXX'或'中标供应商为XXX'，直接回答'XXX'。如果信息中没有供应商信息，请如实告知。
"""
        
        elif question_type == "bid_winner_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接回答中标单位信息。如果信息中有明确的'中标供应商为XXX'，直接回答'XXX'。如果信息中没有中标信息，请如实告知。
"""
        
        elif question_type == "buyer_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接回答采购方信息。如果信息中有明确的'采购方为XXX'，直接回答'XXX'。如果信息中没有采购方信息，请如实告知。
"""
        
        elif question_type == "project_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请从招投标信息中提取项目名称、招标编号、预算金额、中标单位、合同金额、采购方式等关键信息。如果信息不完整，请说明缺少的信息项。
"""
        
        elif question_type == "company_basic_info":
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请回答公司基本信息，包括公司名称、注册地址、经营范围等。如果信息不完整，请如实告知。
"""
        
        else:  # general_info
            instruction = f"""请根据以下招投标信息回答问题：

{context}

问题：{question}

请直接根据信息回答问题，不需要引用法规。
"""
        
        return f"{self.professional_system_prompt}\n\n{instruction}"
    
    def get_simple_prompt(self, context, question):
        """获取简单提示词
"""
        return f"""{context}

问题：{question}

请直接回答："""def get_adaptive_prompt(self, question_type, context, question):
"""
        获取自适应提示词
        """
        # 根据问题类型适度调整提示词
        if question_type in ["company_legal_representative", "price_info", "supplier_info", 
                           "bid_winner_info", "buyer_info", "project_info"]:
            # 招投标相关类型，稍微专业化
            return f"""{self.adaptive_system_prompt}

{context}

问题：{question}

请基于招投标相关信息回答问题："""else:
            # 通用类型，保持简单
            return f
"""{context}

问题：{question}

请根据提供的信息回答问题："""def generate_prompt(self, strategy, question_type, context, question):
"""
        根据策略生成提示词
        """if strategy == "professional":
            return self.get_professional_prompt(question_type, context, question)
        elif strategy == "simple":
            return self.get_simple_prompt(context, question)
        elif strategy == "adaptive":
            return self.get_adaptive_prompt(question_type, context, question)
        else:
            # 默认使用简单提示词
            return self.get_simple_prompt(context, question)

# ====================== 13. 适配GLM3的智能RAG推理函数 ======================
def glm3_smart_rag_inference(tokenizer, glm3_model, embedding_models, index, docs, question, 
                           prompt_selector=None, prompt_generator=None, test_mode="smart_prompt"):
"""
    智能RAG推理函数
    test_mode: "with_prompt", "without_prompt", "smart_prompt"
    """retrieved_docs = enhanced_retrieval_with_reranking(embedding_models, index, docs, question, top_k=8, similarity_threshold=0.03)
    
    if not retrieved_docs:
        retrieved_docs = enhanced_retrieval_with_reranking(embedding_models, index, docs, question, top_k=3, similarity_threshold=0.01)
    
    question_type = analyze_question_type(question)
    
    context = "\n".join([f"相关信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    # 根据测试模式选择提示词策略
    if test_mode == "without_prompt":
        # 不使用提示词模式
        prompt = f"{context}\n\n问题：{question}\n请直接回答："
        strategy_info = {
            "strategy": "no_prompt",
            "reason": "测试模式：不使用提示词"
        }
    
    elif test_mode == "with_prompt":
        # 强制使用专业提示词模式
        if prompt_generator is None:
            prompt_generator = SmartPromptGenerator()
        prompt = prompt_generator.get_professional_prompt(question_type, context, question)
        strategy_info = {
            "strategy": "forced_professional",
            "reason": "测试模式：强制使用专业提示词"
        }
    
    else:  # smart_prompt模式
        # 智能选择提示词策略
        if prompt_selector is None:
            prompt_selector = SmartPromptSelector()
        if prompt_generator is None:
            prompt_generator = SmartPromptGenerator()
            
        strategy_selection = prompt_selector.select_prompt_strategy(question, question_type, retrieved_docs)
        strategy = strategy_selection["strategy"]
        
        prompt = prompt_generator.generate_prompt(strategy, question_type, context, question)
        strategy_info = {
            "strategy": strategy,
            "confidence": strategy_selection["confidence"],
            "reason": strategy_selection["reason"],
            "question_analysis": strategy_selection["question_analysis"],
            "retrieval_analysis": strategy_selection["retrieval_analysis"]
        }
    
    # GLM3推理
"""判断答案是否模糊不清"""ambiguous_keywords = ["无法确定", "不知道", "未提供", "没有提到", "未知", "不清楚", "信息不足"]
    
    # 如果答案是空的或太短
    if not answer or len(answer.strip()) < 3:
        return True
    
    # 如果包含模糊关键词
    if any(keyword in answer for keyword in ambiguous_keywords):
        return True
    
    # 对于特定类型，检查是否包含关键信息
    if question_type == "company_legal_representative":
        if not any(keyword in answer for keyword in ["法定代表人", "法人", "负责人"]):
            return True
    
    elif question_type == "price_info":
        if "元" not in answer and "人民币" not in answer and "￥" not in answer:
            return True
    
    elif question_type == "supplier_info":
        if "供应商" not in answer and "供货" not in answer and "提供" not in answer:
            return True
    
    elif question_type == "bid_winner_info":
        if "中标" not in answer and "成交" not in answer:
            return True
    
    elif question_type == "buyer_info":
        if "采购方" not in answer and "购买方" not in answer and "买方" not in answer:
            return True
    
    elif question_type == "project_info":
        if "项目" not in answer:
            return True
    
    elif question_type == "company_basic_info":
        if "公司" not in answer:
            return True
    
    return False

def post_process_answer(answer, retrieved_docs, question, question_type, test_mode):
"""
    后处理答案，根据测试模式调整
    """
    # 1. 清理答案
    answer = answer.strip()
    
    # 2. 去除可能的特殊标记
    answer = answer.replace("[assistant]", "").replace("[user]", "").replace("[system]", "").strip()
    
    # 3. 根据测试模式调整
    if test_mode == "without_prompt":
        # 无提示词模式，保持原样
        pass
    elif test_mode == "with_prompt":
        # 专业提示词模式，确保专业性
        if "无法确定" in answer or "不知道" in answer:
            if question_type == "price_info":
                answer = "根据《政府采购法》相关规定及现有招投标信息，无法确定具体价格信息。"
            elif question_type == "company_legal_representative":
                answer = "根据《公司法》及招投标文件，法定代表人信息未明确提供。"
    else:  # smart_prompt模式
        # 智能模式，根据答案质量调整
        if len(answer) < 10 and retrieved_docs:
            # 答案太短但有检索结果，尝试从文档提取
            extracted_answer = extract_professional_answer_from_docs(retrieved_docs, question, question_type)
            if extracted_answer:
                answer = f"根据提供的信息：{extracted_answer}"
    
    # 4. 格式化回答（保持简洁）
    if answer and answer[-1] not in ['。', '！', '？', '.', '!', '?']:
        answer += '。'
    
    return answer

def extract_professional_answer_from_docs(docs, question, question_type):
    """从检索文档中直接提取答案
"""
    # 针对法定代表人问题
    if question_type == "company_legal_representative":
        for doc in docs:
            patterns = [
                r'法定代表人是([\u4e00-\u9fa5]{2,4})',
                r'法人代表是([\u4e00-\u9fa5]{2,4})',
                r'负责人是([\u4e00-\u9fa5]{2,4})',
                r'法定代表人：([\u4e00-\u9fa5]{2,4})',
                r'法人代表：([\u4e00-\u9fa5]{2,4})'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc)
                if match:
                    return match.group(1)
    
    # 针对价格问题
    elif question_type == "price_info":
        for doc in docs:
            patterns = [
                r'价格为?(\d+\.?\d*)[万万千]?元',
                r'金额为?(\d+\.?\d*)[万万千]?元',
                r'成交价[为是]?(\d+\.?\d*)[万万千]?元',
                r'(\d+\.?\d*)[万万千]?元'
            ]
            for pattern in patterns:
                matches = re.findall(pattern, doc)
                if matches:
                    price = matches[0]
                    return f"{price}元"
    
    # 针对供应商问题
    elif question_type == "supplier_info":
        for doc in docs:
            patterns = [
                r'供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'由([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供',
                r'供货商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc)
                if match:
                    return match.group(1)
    
    # 针对中标方问题
    elif question_type == "bid_winner_info":
        for doc in docs:
            patterns = [
                r'中标供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'中标方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'中标公司是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'中标单位为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc)
                if match:
                    return match.group(1)
    
    # 针对采购方问题
    elif question_type == "buyer_info":
        for doc in docs:
            patterns = [
                r'采购方为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'采购方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'购买方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
                r'买方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
            ]
            for pattern in patterns:
                match = re.search(pattern, doc)
                if match:
                    return match.group(1)
    
    # 针对项目信息
    elif question_type == "project_info":
        for doc in docs:
            if "项目名称" in doc:
                match = re.search(r'项目名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})', doc)
                if match:
                    return match.group(1)
    
    # 针对公司基本信息
    elif question_type == "company_basic_info":
        for doc in docs:
            if "公司名称" in doc:
                match = re.search(r'公司名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})', doc)
                if match:
                    return match.group(1)
    
    return None

# ====================== 14. 构建FAISS向量索引 ======================
def build_optimized_vector_index(embedding_models, docs, batch_size=32):
    doc_vectors = []
    valid_docs = []
    
    print(f"正在编码 {len(docs)} 条文档...")
    
    # 过滤掉过短的文档，提高索引质量
    filtered_docs = [doc for doc in docs if len(doc.strip()) > 5]  # 降低长度要求
    print(f"  过滤后有效文档数量: {len(filtered_docs)}")
    
    for i in range(0, len(filtered_docs), batch_size):
        batch_docs = filtered_docs[i:i+batch_size]
        try:
            # 调用修复后的批量编码函数
            batch_vectors = bge_embedding_encode(embedding_models, batch_docs, batch_mode=True)
            
            # 确保batch_vectors是有效的numpy数组且不为空
            if isinstance(batch_vectors, np.ndarray) and batch_vectors.shape[0] > 0:
                # 处理批量返回的向量
                for j, vec in enumerate(batch_vectors):
                    # 检查向量是否有效（非零向量且维度正确）
                    if isinstance(vec, np.ndarray) and vec.size > 0 and np.linalg.norm(vec) > 0:
                        doc_vectors.append(vec)
                        # 确保索引不越界
                        if j < len(batch_docs):
                            valid_docs.append(batch_docs[j])
            else:
                # 批量编码失败，尝试逐文档编码
                print(f"  批次 {i//batch_size + 1} 批量编码返回空结果，尝试逐文档编码")
                for doc in batch_docs:
                    try:
                        vec = bge_embedding_encode(embedding_models, doc, batch_mode=False)
                        if isinstance(vec, np.ndarray) and vec.size > 0 and np.linalg.norm(vec) > 0:
                            doc_vectors.append(vec)
                            valid_docs.append(doc)
                    except Exception as doc_e:
                        print(f"    文档编码失败: {doc_e}")
        except Exception as e:
            # 批量编码异常，尝试逐文档编码
            print(f"  批次 {i//batch_size + 1} 批量编码失败，尝试逐文档编码: {e}")
            for doc in batch_docs:
                try:
                    vec = bge_embedding_encode(embedding_models, doc, batch_mode=False)
                    if isinstance(vec, np.ndarray) and vec.size > 0 and np.linalg.norm(vec) > 0:
                        doc_vectors.append(vec)
                        valid_docs.append(doc)
                except Exception as doc_e:
                    print(f"    文档编码失败: {doc_e}")
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已编码 {min(i+batch_size, len(filtered_docs))}/{len(filtered_docs)} 条文档")
    
    if not doc_vectors:
        print("❌ 无有效文档向量，无法构建FAISS索引")
        # 紧急处理：如果没有有效向量，直接返回原始文档作为备选
        return None, filtered_docs
    
    # 确保所有向量维度一致
    vector_dim = doc_vectors[0].shape[0]
    consistent_vectors = []
    consistent_docs = []
    for vec, doc in zip(doc_vectors, valid_docs):
        if vec.shape[0] == vector_dim:
            consistent_vectors.append(vec)
            consistent_docs.append(doc)
    
    doc_vectors = np.array(consistent_vectors, dtype=np.float32)
    valid_docs = consistent_docs
    
    print(f"  最终有效向量数量: {len(doc_vectors)}")
    
    # 使用更高效的索引类型
    try:
        # 对于大规模文档，使用IVF索引
        if len(doc_vectors) > 1000:
            nlist = min(100, len(doc_vectors) // 10)
            quantizer = faiss.IndexFlatIP(vector_dim)
            index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(doc_vectors)
            index.add(doc_vectors)
            index.nprobe = min(10, nlist // 2)
            index_type = "IndexIVFFlat"
        else:
            # 小规模文档，使用Flat索引
            index = faiss.IndexFlatIP(vector_dim)
            index.add(doc_vectors)
            index_type = "IndexFlatIP"
        
        print(f"✅ 共构建 {len(valid_docs)} 条有效文档的FAISS索引")
        print(f"  向量维度: {vector_dim}")
        print(f"  索引类型: {index_type} (内积相似度)")
        print(f"  索引构建成功率: {len(valid_docs)/len(filtered_docs)*100:.2f}%")
        
        return index, valid_docs
    except Exception as e:
        print(f"❌ FAISS索引构建失败，使用备选方案: {e}")
        # 备选方案：返回所有文档，不使用向量索引
        return None, filtered_docs

# ====================== 15. 加载问答对 ======================
def load_project_qa_data(qa_file_path="qa_data/520_qa.json", kb_file_path="qa_data/knowledge_base.txt"):
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

# ====================== 16. 评估函数 ======================
def calculate_recall(retrieved_docs, relevant_docs):
    """计算召回率：检索到的文档中是否包含至少一个相关文档
"""
    if not retrieved_docs or not relevant_docs:
        return 0
    
    for retrieved_doc in retrieved_docs:
        for relevant_doc in relevant_docs:
            if is_doc_related(retrieved_doc, relevant_doc):
                return 1
    
    return 0

def is_doc_related(doc1, doc2):
    """判断两个文档是否相关
"""
    # 直接包含关系
    if doc1 in doc2 or doc2 in doc1:
        return True
    
    # 提取关键实体比较
    entities1 = extract_entities_from_text(doc1)
    entities2 = extract_entities_from_text(doc2)
    
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    # 文本相似度
    similarity = SequenceMatcher(None, doc1, doc2).ratio()
    return similarity > 0.5  # 降低阈值

def extract_entities_from_text(text):
    """从文本中提取实体
"""
    entities = []
    
    # 公司名称
    company_patterns = [
        r'([\u4e00-\u9fa5a-zA-Z0-9]{2,})(?:有限公司|公司|集团)',
        r'供应商[：:]?([\u4e00-\u9fa5a-zA-Z0-9]{2,})',
        r'采购方[：:]?([\u4e00-\u9fa5a-zA-Z0-9]{2,})',
        r'中标方[：:]?([\u4e00-\u9fa5a-zA-Z0-9]{2,})',
        r'由([\u4e00-\u9fa5a-zA-Z0-9]{2,})提供'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # 人名
    name_patterns = [
        r'法定代表人是([\u4e00-\u9fa5]{2,4})',
        r'法人代表是([\u4e00-\u9fa5]{2,4})'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # 价格
    price_matches = re.findall(r'(\d+\.?\d*)[万万千]?元', text)
    entities.extend([f"{price}元" for price in price_matches])
    
    # 项目名称
    project_matches = re.findall(r'项目名称[：:]?([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})', text)
    entities.extend(project_matches)
    
    return list(set(entities))

def enhanced_accuracy_calculation(model_answer, reference_answer, question):
    """增强版准确率计算，考虑更多匹配情况
"""
    if not model_answer or not reference_answer:
        return 0
    
    # 处理"未知"情况
    unknown_keywords = ["未知", "无法确定", "不能确定", "不清楚", "不明确", "暂无", "信息未知"]
    
    # 如果标准答案是"未知"，模型回答也应该是"未知"或类似
    if any(keyword in reference_answer for keyword in unknown_keywords):
        if any(keyword in model_answer for keyword in unknown_keywords):
            return 1
    
    # 简单完全匹配
    if reference_answer in model_answer or model_answer in reference_answer:
        return 1
    
    # 忽略标点和空格后的匹配
    ref_clean = re.sub(r'[，。；：、\s]', '', reference_answer)
    model_clean = re.sub(r'[，。；：、\s]', '', model_answer)
    
    if ref_clean in model_clean or model_clean in ref_clean:
        return 1
    
    # 提取关键信息比较
    ref_info = extract_key_info(reference_answer, question)
    model_info = extract_key_info(model_answer, question)
    
    if not ref_info:
        # 如果没有提取到关键信息，使用文本相似度
        similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
        return 1 if similarity > 0.7 else 0  # 恢复0.7阈值
    
    # 比较关键信息
    match_count = 0
    for key, value in ref_info.items():
        if key in model_info:
            # 检查值是否匹配
            if value == model_info[key] or value in model_info[key] or model_info[key] in value:
                match_count += 1
    
    if match_count >= len(ref_info) * 0.7:  # 恢复70%阈值
        return 1
    
    # 编辑距离作为最后手段
    similarity = SequenceMatcher(None, model_answer, reference_answer).ratio()
    return 1 if similarity > 0.7 else 0  # 恢复0.7阈值

def extract_key_info(text, question):
    """根据问题类型提取关键信息
"""
    info = {}
    
    # 法定代表人
    if "法定代表人" in question or "法人" in question:
        patterns = [r'法定代表人是([\u4e00-\u9fa5]{2,4})', r'法人代表是([\u4e00-\u9fa5]{2,4})']
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                info["法定代表人"] = match.group(1)
                break
    
    # 价格
    elif "价格" in question or "多少钱" in question or "金额" in question:
        patterns = [r'(\d+\.?\d*)[万万千]?元', r'价格[为是]?(\d+\.?\d*)[万万千]?元']
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                info["价格"] = f"{matches[0]}元"
                break
    
    # 供应商
    elif "供应商" in question or "提供" in question:
        patterns = [
            r'供应商是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
            r'供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
            r'由([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})提供'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                info["供应商"] = match.group(1)
                break
    
    # 采购方
    elif "采购方" in question or "购买" in question:
        patterns = [
            r'采购方为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
            r'采购方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                info["采购方"] = match.group(1)
                break
    
    # 中标方
    elif "中标" in question:
        patterns = [
            r'中标供应商为([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})',
            r'中标方是([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                info["中标方"] = match.group(1)
                break
    
    # 项目信息
    elif "项目" in question:
        match = re.search(r'项目名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()《》\-]{4,})', text)
        if match:
            info["项目名称"] = match.group(1)
    
    # 公司信息
    elif "公司" in question and ("名称" in question or "基本信息" in question):
        match = re.search(r'公司名称[：:为是]*([\u4e00-\u9fa5a-zA-Z0-9（）()]{4,})', text)
        if match:
            info["公司名称"] = match.group(1)
    
    return info

# ====================== 17. GLM3模型智能测试流程 ======================
def run_glm3_smart_test(test_mode="smart_prompt"):
    """GLM3模型智能测试流程
    test_mode: "with_prompt"
"""
    # 初始化问题类型统计（每个测试模式单独统计）
    question_type_stats = {
        "company_legal_representative": {"total": 0, "correct": 0, "recall": 0},
        "price_info": {"total": 0, "correct": 0, "recall": 0},
        "supplier_info": {"total": 0, "correct": 0, "recall": 0},
        "buyer_info": {"total": 0, "correct": 0, "recall": 0},
        "bid_winner_info": {"total": 0, "correct": 0, "recall": 0},
        "project_info": {"total": 0, "correct": 0, "recall": 0},
        "company_basic_info": {"total": 0, "correct": 0, "recall": 0},
        "general_info": {"total": 0, "correct": 0, "recall": 0}
    }
    
    # 策略使用统计
    strategy_stats = {
        "professional": {"count": 0, "correct": 0},
        "simple": {"count": 0, "correct": 0},
        "adaptive": {"count": 0, "correct": 0},
        "no_prompt": {"count": 0, "correct": 0},
        "forced_professional": {"count": 0, "correct": 0}
    }
    
    # 初始化目录
    output_dir, tb_log_dir, result_dir, log_dir = init_output_dir(test_mode)
    
    log_file_path = os.path.join(log_dir, f"glm3_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = LoggerRedirect(log_file_path)
    tb_writer = init_tensorboard(tb_log_dir)
    
    # GLM3模型配置
    test_config = {
        "llm_name": "chatglm3-6b",
        "glm3_local_path": "/mnt/workspace/data/modelscope/cache/ZhipuAI/chatglm3-6b",
        "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-
"""对比三种测试模式的结果
"""
    print("\n" + "="*80)
    print("📊 三种测试模式结果对比")
    print("="*80)
    
    if not with_prompt_results or not without_prompt_results or not smart_prompt_results:
        print("❌ 无法完整对比，因为至少有一种测试模式没有结果")
        return
    
    # 计算总体指标
    with_prompt_recall = sum([r["recall_score"] for r in with_prompt_results]) / len(with_prompt_results)
    with_prompt_accuracy = sum([r["accuracy"] for r in with_prompt_results]) / len(with_prompt_results)
    
    without_prompt_recall = sum([r["recall_score"] for r in without_prompt_results]) / len(without_prompt_results)
    without_prompt_accuracy = sum([r["accuracy"] for r in without_prompt_results]) / len(without_prompt_results)
    
    smart_prompt_recall = sum([r["recall_score"] for r in smart_prompt_results]) / len(smart_prompt_results)
    smart_prompt_accuracy = sum([r["accuracy"] for r in smart_prompt_results]) / len(smart_prompt_results)
    
    print(f"\n📈 总体指标对比：")
    print(f"  指标\t\t\t无提示词\t专业提示词\t智能提示词\t最佳提升")
    print(f"  {'-'*80}")
    
    recall_improvement = smart_prompt_recall - max(with_prompt_recall, without_prompt_recall)
    accuracy_improvement = smart_prompt_accuracy - max(with_prompt_accuracy, without_prompt_accuracy)
    
    print(f"  平均召回率\t\t{without_prompt_recall:.4f}\t\t{with_prompt_recall:.4f}\t\t{smart_prompt_recall:.4f}\t\t{recall_improvement:+.4f}")
    print(f"  平均准确率\t\t{without_prompt_accuracy:.4f}\t\t{with_prompt_accuracy:.4f}\t\t{smart_prompt_accuracy:.4f}\t\t{accuracy_improvement:+.4f}")
    
    # 按问题类型对比准确率
    print(f"\n🔍 按问题类型准确率对比：")
    print(f"  问题类型\t\t\t无提示词\t专业提示词\t智能提示词\t智能vs最佳")
    print(f"  {'-'*80}")
    
    # 收集三种模式的问题类型统计
    results_by_mode = {
        "without": without_prompt_results,
        "with": with_prompt_results,
        "smart": smart_prompt_results
    }
    
    type_stats = {}
    for mode, results in results_by_mode.items():
        for result in results:
            q_type = result["question_type"]
            if q_type not in type_stats:
                type_stats[q_type] = {"without": {"total": 0, "correct": 0},
                                      "with": {"total": 0, "correct": 0},
                                      "smart": {"total": 0, "correct": 0}}
            
            type_stats[q_type][mode]["total"] += 1
            if result["accuracy"] == 1:
                type_stats[q_type][mode]["correct"] += 1
    
    # 计算并打印每个问题类型的准确率
    for q_type in sorted(type_stats.keys()):
        without_acc = type_stats[q_type]["without"]["correct"] / type_stats[q_type]["without"]["total"] if type_stats[q_type]["without"]["total"] > 0 else 0
        with_acc = type_stats[q_type]["with"]["correct"] / type_stats[q_type]["with"]["total"] if type_stats[q_type]["with"]["total"] > 0 else 0
        smart_acc = type_stats[q_type]["smart"]["correct"] / type_stats[q_type]["smart"]["total"] if type_stats[q_type]["smart"]["total"] > 0 else 0
        
        # 计算智能策略相对于最佳基线（无提示词或专业提示词）的提升
        best_baseline = max(without_acc, with_acc)
        improvement = smart_acc - best_baseline if best_baseline > 0 else 0
        
        print(f"  {q_type:<25} {without_acc:>6.1%}\t\t{with_acc:>6.1%}\t\t{smart_acc:>6.1%}\t\t{improvement:>+7.1%}")
    
    # 智能策略分析
    print(f"\n🤖 智能提示词策略分析：")
    
    # 收集智能模式下的策略使用情况
    strategy_performance = {}
    for result in smart_prompt_results:
        strategy = result["prompt_strategy"]["strategy"]
        if strategy not in strategy_performance:
            strategy_performance[strategy] = {"total": 0, "correct": 0}
        
        strategy_performance[strategy]["total"] += 1
        if result["accuracy"] == 1:
            strategy_performance[strategy]["correct"] += 1
    
    for strategy, stats in strategy_performance.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            print(f"  {strategy:<15}: {stats['total']:>3}次 ({stats['total']/len(smart_prompt_results)*100:5.1f}%)，准确率：{accuracy:.1%}")
    
    # 保存对比结果
    comparison_dir = "./output_comparison"
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    comparison_file = os.path.join(comparison_dir, f"comparison_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(comparison_file, "w", encoding="utf-8") as f:
        f.write("GLM3三种测试模式结果对比\n")
        f.write("="*80 + "\n")
        f.write(f"对比时间：{datetime.now().strftime('%Y-%m-d %H:%M:%S')}\n")
        f.write(f"测试用例总数：{len(smart_prompt_results)}\n\n")
        
        f.write("总体指标对比：\n")
        f.write(f"  平均召回率：无提示词={without_prompt_recall:.4f}, 专业提示词={with_prompt_recall:.4f}, 智能提示词={smart_prompt_recall:.4f}\n")
        f.write(f"  平均准确率：无提示词={without_prompt_accuracy:.4f}, 专业提示词={with_prompt_accuracy:.4f}, 智能提示词={smart_prompt_accuracy:.4f}\n")
        f.write(f"  智能提示词提升：召回率{recall_improvement:+.4f}, 准确率{accuracy_improvement:+.4f}\n\n")
        
        f.write("按问题类型准确率对比：\n")
        for q_type in sorted(type_stats.keys()):
            without_acc = type_stats[q_type]["without"]["correct"] / type_stats[q_type]["without"]["total"] if type_stats[q_type]["without"]["total"] > 0 else 0
            with_acc = type_stats[q_type]["with"]["correct"] / type_stats[q_type]["with"]["total"] if type_stats[q_type]["with"]["total"] > 0 else 0
            smart_acc = type_stats[q_type]["smart"]["correct"] / type_stats[q_type]["smart"]["total"] if type_stats[q_type]["smart"]["total"] > 0 else 0
            
            best_baseline = max(without_acc, with_acc)
            improvement = smart_acc - best_baseline if best_baseline > 0 else 0
            
            f.write(f"  {q_type:<25}: 无提示词={without_acc:.1%}, 专业提示词={with_acc:.1%}, 智能提示词={smart_acc:.1%}, 提升={improvement:+.1%}\n")
        
        f.write("\n智能提示词策略分析：\n")
        for strategy, stats in strategy_performance.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                f.write(f"  {strategy:<15}: {stats['total']:>3}次 ({stats['total']/len(smart_prompt_results)*100:5.1f}%)，准确率：{accuracy:.1%}\n")
    
    print(f"\n✅ 对比结果已保存到：{comparison_file}")
    print("="*80)

# ====================== 19. 主函数 ======================
if __name__ == "__main__":
    print("="*80)
    print("GLM3模型智能提示词测试系统")
    print("主要功能：")
    print("1. 模式A：使用专业提示词模板进行测试")
    print("2. 模式B：不使用任何提示词进行基线测试")
    print("3. 模式C：使用智能提示词选择策略进行测试")
    print("4. 自动对比三种模式的测试结果")
    print("="*80)
    
    print("\n请选择测试模式：")
    print("1. 仅运行'使用提示词模板'测试")
    print("2. 仅运行'不使用提示词模板'测试")
    print("3. 仅运行'智能提示词选择'测试")
    print("4. 运行三种测试并进行全面对比")
    
    choice = input("请输入选择（1/2/3/4，默认4）：").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("开始运行'使用提示词模板'测试")
        print("="*60)
        with_prompt_results = run_glm3_smart_test(test_mode="with_prompt")
        
    elif choice == "2":
        print("\n" + "="*60)
        print("开始运行'不使用提示词模板'测试")
        print("="*60)
        without_prompt_results = run_glm3_smart_test(test_mode="without_prompt")
        
    elif choice == "3":
        print("\n" + "="*60)
        print("开始运行'智能提示词选择'测试")
        print("="*60)
        smart_prompt_results = run_glm3_smart_test(test_mode="smart_prompt")
        
    else:  # 默认选择4
        print("\n" + "="*60)
        print("开始运行'使用提示词模板'测试")
        print("="*60)
        with_prompt_results = run_glm3_smart_test(test_mode="with_prompt")
        
        print("\n" + "="*60)
        print("开始运行'不使用提示词模板'测试")
        print("="*60)
        without_prompt_results = run_glm3_smart_test(test_mode="without_prompt")
        
        print("\n" + "="*60)
        print("开始运行'智能提示词选择'测试")
        print("="*60)
        smart_prompt_results = run_glm3_smart_test(test_mode="smart_prompt")
        
        # 对比三种测试模式的结果
        if with_prompt_results and without_prompt_results and smart_prompt_results:
            compare_all_test_results(with_prompt_results, without_prompt_results, smart_prompt_results)
    
    print("\n🎉 所有测试任务已完成！")