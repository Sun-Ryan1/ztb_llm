import torch
import json
import numpy as np
import faiss
import os
import re
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import sys

# ====================== 配置类 ======================
class ModelConfig:
    """模型配置类"""
    def __init__(self, llm_name, llm_local_path, embedding_local_path):
        self.llm_name = llm_name
        self.llm_local_path = llm_local_path
        self.embedding_local_path = embedding_local_path

# ====================== GLM3模型配置 ======================
class GLM3TestConfig:
    """GLM3模型测试配置"""
    def __init__(self):
        # GLM3模型配置
        self.llm_config = {
            "llm_name": "ChatGLM3-6B",
            "llm_local_path": "/mnt/workspace/data/modelscope/cache/ZhipuAI/chatglm3-6b",
            "embedding_local_path": "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5/BAAI/bge-large-zh-v1___5"
        }
        
        # 测试数据路径
        self.test_data_path = "qa_data/100_qa.json"
        self.knowledge_base_path = "qa_data/knowledge_base.txt"
        
        # 输出目录（包含模型名称）
        self.output_dir = f"./glm3_model_results"
        
        # 测试参数
        self.max_test_cases = 50
        self.batch_size = 16
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.75

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
        print(f"正在加载GLM3模型：{config['llm_name']}")
        
        try:
            # 1. 加载GLM3模型tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                padding_side="left",  # GLM3需要左侧填充
                local_files_only=True
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ GLM3 Tokenizer加载成功")
            print(f"   pad_token: {tokenizer.pad_token}")
            print(f"   eos_token: {tokenizer.eos_token}")
            
            # 2. 加载GLM3模型
            llm_model = AutoModelForCausalLM.from_pretrained(
                config["llm_local_path"],
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,  # GLM3使用float16
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            llm_model.eval()
            print(f"✅ GLM3模型 {config['llm_name']} 加载成功")
        except Exception as e:
            print(f"❌ GLM3模型加载失败，错误信息：{e}")
            import traceback
            traceback.print_exc()
            return None, None, None
        
            # 3. 加载Embedding模型
        print(f"正在加载BGE Embedding模型")
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                local_files_only=True
            )
    
            # 修改这里：移除 device_map='auto'，改为手动将模型移到GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding_model = AutoModel.from_pretrained(
                config["embedding_local_path"],
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
            ).to(device)  # 手动将模型移到设备
    
            embedding_model.eval()
            print(f"✅ BGE Embedding模型加载成功")
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

# ====================== BGE文本转向量函数 ======================
def bge_embedding_encode(embedding_models, text, batch_mode=False):
    """BGE文本转向量函数"""
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
    """构建FAISS向量索引"""
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
    """改进的检索函数"""
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
        "地址", "注册地址", "经营地址"
    ]
    
    for keyword in bid_keywords:
        if keyword in text:
            keywords.append(keyword)
    
    return keywords

# ====================== 推理函数 ======================
def direct_inference_no_prompt(tokenizer, glm3_model, question):
    """场景1：无任何提示词，直接让模型回答问题"""
    # GLM3需要构建对话格式
    formatted_prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to("cuda")
    
    # 确保有pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = glm3_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant的回答
    model_answer = ""
    if "<|assistant|>" in full_response:
        parts = full_response.split("<|assistant|>")
        if len(parts) > 1:
            model_answer = parts[-1].strip()
    elif formatted_prompt in full_response:
        model_answer = full_response[len(formatted_prompt):].strip()
    else:
        model_answer = full_response.strip()
    
    # 清理可能的特殊标记
    model_answer = model_answer.replace("[assistant]", "").replace("[user]", "").replace("[system]", "").strip()
    
    return model_answer, []  # 无检索文档

def optimized_rag_inference(tokenizer, glm3_model, embedding_models, index, docs, question):
    """场景2：使用提示词模板的RAG推理"""
    retrieved_docs = enhanced_retrieval(embedding_models, index, docs, question, top_k=5, similarity_threshold=0.75)
    
    if not retrieved_docs:
        return "根据现有信息无法确定。", []
    
    context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    # 使用上传文件中的专业提示词模板
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

问：{question}
答："""
    
    # GLM3对话格式
    formatted_prompt = f"<|system|>\n你是聚焦招投标采购全流程的专业智能问答系统。\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    with torch.no_grad():
        outputs = glm3_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant的回答
    model_answer = ""
    if "<|assistant|>" in full_response:
        parts = full_response.split("<|assistant|>")
        if len(parts) > 1:
            model_answer = parts[-1].strip()
    elif formatted_prompt in full_response:
        model_answer = full_response[len(formatted_prompt):].strip()
    else:
        model_answer = full_response.strip()
    
    # 清理可能的特殊标记
    model_answer = model_answer.replace("[assistant]", "").replace("[user]", "").replace("[system]", "").strip()
    
    return model_answer, retrieved_docs

# ====================== 数据加载器 ======================
def load_qa_data(qa_file_path="qa_data/100_qa.json", kb_file_path="qa_data/knowledge_base.txt"):
    """加载QA数据"""
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
    
    ref_entities = extract_entities_from_text(reference_answer)
    model_entities = extract_entities_from_text(model_answer)
    
    if ref_entities:
        common_entities = set(ref_entities) & set(model_entities)
        if len(common_entities) / len(ref_entities) >= 0.5:
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
class GLM3ModelTestRunner:
    """GLM3模型测试运行器"""
    
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
            print(f"📊 限制测试用例数为: {len(self.test_cases)}")
        
        print(f"\n{'='*60}")
        print(f"GLM3模型测试初始化完成")
        print(f"模型名称: {config.llm_config['llm_name']}")
        print(f"测试用例数: {len(self.test_cases)}")
        print(f"知识库文档数: {len(self.knowledge_docs)}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _setup_output_dir(self):
        """设置输出目录"""
        # 使用模型名称和时间戳创建唯一目录
        model_name = self.config.llm_config['llm_name'].replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.config.output_dir, f"{model_name}_{timestamp}")
        
        # 只创建llm_results和logs目录（根据要求删除对比相关目录）
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "llm_results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        return output_dir
    
    def run_glm3_tests(self):
        """运行GLM3模型测试"""
        print(f"\n{'='*60}")
        print(f"开始GLM3模型测试")
        print(f"模型: {self.config.llm_config['llm_name']}")
        print(f"{'='*60}")
        
        # 加载GLM3模型
        tokenizer, glm3_model, embedding_models = ModelManager.load_local_models(self.config.llm_config)
        if glm3_model is None:
            print(f"❌ GLM3模型加载失败，测试终止")
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
            print(f"问题：{question}")
            
            # 场景1：无提示词直接回答
            print("\n--- 场景1：无提示词 ---")
            try:
                model_answer1, _ = direct_inference_no_prompt(tokenizer, glm3_model, question)
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
                print(f"❌ 场景1测试失败：{e}")
                scenario1_results.append({
                    "scenario": "no_prompt",
                    "test_case_id": idx + 1,
                    "question": question,
                    "error": str(e)
                })
            
            # 场景2：有提示词的RAG回答
            print("\n--- 场景2：有提示词RAG ---")
            if index is not None:
                try:
                    model_answer2, retrieved_docs = optimized_rag_inference(
                        tokenizer, glm3_model, embedding_models, index, enhanced_docs, question
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
                    print(f"❌ 场景2测试失败：{e}")
                    scenario2_results.append({
                        "scenario": "with_prompt_rag",
                        "test_case_id": idx + 1,
                        "question": question,
                        "error": str(e)
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
                    avg_acc1 = sum([r.get("accuracy", 0) for r in scenario1_results if "accuracy" in r]) / len([r for r in scenario1_results if "accuracy" in r])
                    print(f"\n📊 当前进度：{idx+1}/{len(self.test_cases)}")
                    print(f"  场景1平均准确率：{avg_acc1:.4f}")
                
                if scenario2_results and any("accuracy" in r for r in scenario2_results):
                    valid_results2 = [r for r in scenario2_results if "accuracy" in r]
                    if valid_results2:
                        avg_acc2 = sum([r["accuracy"] for r in valid_results2]) / len(valid_results2)
                        avg_recall2 = sum([r.get("recall_score", 0) for r in valid_results2]) / len(valid_results2)
                        print(f"  场景2平均准确率：{avg_acc2:.4f}，平均召回率：{avg_recall2:.4f}")
        
        # 保存GLM3模型的测试结果
        glm3_results = {
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
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果文件
        result_file = os.path.join(self.output_dir, "llm_results", 
                                  f"glm3_test_results_{datetime.now().strftime('%H%M%S')}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(glm3_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ GLM3模型测试完成")
        print(f"   场景1平均准确率: {glm3_results['summary']['scenario1_avg_accuracy']:.4f}")
        print(f"   场景2平均准确率: {glm3_results['summary']['scenario2_avg_accuracy']:.4f}")
        print(f"   场景2平均召回率: {glm3_results['summary']['scenario2_avg_recall']:.4f}")
        print(f"   结果文件: {result_file}")
        
        # 生成日志文件
        self._generate_log_file(glm3_results)
        
        # 清理资源
        ModelManager.cleanup_models(glm3_model, embedding_models)
        if index is not None:
            del index
        if enhanced_docs is not None:
            del enhanced_docs
        
        print(f"\n{'='*60}")
        print("GLM3模型测试完成")
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
    
    def _generate_log_file(self, glm3_results):
        """生成日志文件"""
        log_content = f"""GLM3模型测试日志
测试时间: {glm3_results['test_time']}
模型名称: {glm3_results['llm_config']['llm_name']}
模型路径: {glm3_results['llm_config']['llm_local_path']}

测试配置:
  测试用例数: {glm3_results['test_config']['test_cases_count']}
  知识库文档数: {glm3_results['test_config']['knowledge_docs_count']}
  最大测试用例数: {glm3_results['test_config']['max_test_cases']}
  批处理大小: {glm3_results['test_config']['batch_size']}
  检索top_k: {glm3_results['test_config']['top_k_retrieval']}
  相似度阈值: {glm3_results['test_config']['similarity_threshold']}

测试结果摘要:
  场景1平均准确率: {glm3_results['summary']['scenario1_avg_accuracy']:.4f}
  场景1平均质量分: {glm3_results['summary']['scenario1_avg_quality']:.4f}
  场景1平均回答长度: {glm3_results['summary']['scenario1_avg_answer_length']:.2f}
  场景2平均准确率: {glm3_results['summary']['scenario2_avg_accuracy']:.4f}
  场景2平均召回率: {glm3_results['summary']['scenario2_avg_recall']:.4f}
  场景2平均质量分: {glm3_results['summary']['scenario2_avg_quality']:.4f}
  场景2平均回答长度: {glm3_results['summary']['scenario2_avg_answer_length']:.2f}
  场景2平均检索文档数: {glm3_results['summary']['scenario2_avg_retrieved_count']:.2f}

详细结果请查看llm_results目录下的JSON文件。
"""
        
        log_file = os.path.join(self.output_dir, "logs", "test_summary.log")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)
        
        print(f"✅ 日志文件已生成: {log_file}")

# ====================== 主函数 ======================
def main():
    """主函数：运行GLM3模型测试"""
    
    print(f"{'='*60}")
    print("GLM3模型测试")
    print("测试场景：")
    print("  1. 无提示词直接推理（测试学习能力）")
    print("  2. 有提示词RAG推理（测试可训练能力）")
    print(f"{'='*60}")
    
    # 创建GLM3测试配置
    test_config = GLM3TestConfig()
    
    # 创建测试运行器
    test_runner = GLM3ModelTestRunner(test_config)
    
    # 运行GLM3模型测试
    test_runner.run_glm3_tests()
    
    print(f"\n{'='*60}")
    print("✅ GLM3模型测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()