import torch
import json
import numpy as np
import faiss
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
    """记录TensorBoard日志，区分场景"""
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"{metric_name}/{scenario}", metric_value, step)

# ---------------------- 4. 配置量化参数 ----------------------
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# ---------------------- 5. 加载本地模型 ----------------------
def load_local_models(llm_name, llm_local_path, embedding_local_path):
    print(f"正在加载本地大模型：{llm_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_local_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_local_path,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=get_bnb_config(),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        llm_model.eval()
        print(f"✅ 大模型 {llm_name} 加载成功")
    except Exception as e:
        print(f"❌ 大模型加载失败，错误信息：{e}")
        return None, None, None
    
    print(f"正在加载本地BGE Embedding模型：{embedding_local_path.split('/')[-1]}")
    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_local_path,
            trust_remote_code=True,
            local_files_only=True
        )
        embedding_model = AutoModel.from_pretrained(
            embedding_local_path,
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

# ---------------------- 6. BGE文本转向量函数 ----------------------
def bge_embedding_encode(embedding_models, text, batch_mode=False):
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

# ---------------------- 7. 构建FAISS向量索引 ----------------------
def build_optimized_vector_index(embedding_models, docs, batch_size=32):
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

# ---------------------- 8. 改进的检索函数 ----------------------
def enhanced_retrieval(embedding_models, index, docs, question, top_k=10, similarity_threshold=0.15):
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

# ---------------------- 9. RAG推理函数（有提示词） ----------------------
def optimized_rag_inference(tokenizer, llm_model, embedding_models, index, docs, question):
    """场景2：使用提示词模板的RAG推理"""
    retrieved_docs = enhanced_retrieval(embedding_models, index, docs, question, top_k=10, similarity_threshold=0.15)
    
    if not retrieved_docs:
        return "根据现有信息无法确定。", []
    
    context = "\n".join([f"信息{i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""你是一个招投标领域的助手。请根据以下信息回答问题：

{context}

问题：{question}

要求：
1. 只基于上述信息回答
2. 如果信息足够，请直接给出答案
3. 如果信息不足，请说"根据现有信息无法确定"
4. 答案要简洁准确

回答："""
    
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
    
    return model_answer, retrieved_docs

# ---------------------- 10. 直接推理函数（无提示词） ----------------------
def direct_inference_no_prompt(tokenizer, llm_model, question):
    """场景1：无任何提示词，直接让模型回答问题"""
    
    inputs = tokenizer(
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

# ---------------------- 11. 加载问答对（新版本） ----------------------
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

# ---------------------- 12. 评估函数 ----------------------
def calculate_recall(retrieved_docs, relevant_docs):
    if not retrieved_docs or not relevant_docs:
        return 0
    
    for retrieved_doc in retrieved_docs:
        for relevant_doc in relevant_docs:
            if is_doc_related(retrieved_doc, relevant_doc):
                return 1
    
    return 0

def is_doc_related(doc1, doc2):
    entities1 = extract_entities_from_text(doc1)
    entities2 = extract_entities_from_text(doc2)
    
    common_entities = set(entities1) & set(entities2)
    if common_entities:
        return True
    
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, doc1, doc2).ratio()
    return similarity > 0.6

def extract_entities_from_text(text):
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

# ---------------------- 13. 双场景测试流程 ----------------------
def run_dual_scenario_test():
    """运行双场景测试：无提示词 vs 有提示词RAG"""
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
    
    # 3. 构建FAISS索引（仅用于场景2）
    print("\n正在构建FAISS向量索引（用于场景2 RAG）...")
    index, enhanced_docs = build_optimized_vector_index(embedding_models, test_docs, batch_size=16)
    if index is None:
        print("\n❌ FAISS索引构建失败，场景2将无法测试")
        # 不终止，继续运行场景1
    
    print("✅ FAISS索引构建完成")
    
    # 4. 执行双场景测试
    test_results_scenario1 = []  # 场景1：无提示词
    test_results_scenario2 = []  # 场景2：有提示词RAG
    
    print("\n" + "="*60)
    print("开始双场景模型能力测试...")
    print("场景1：无提示词直接生成")
    print("场景2：有提示词RAG生成")
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
        
        # 场景2：有提示词的RAG回答
        print("\n--- 场景2：有提示词RAG ---")
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
                    "retrieved_docs": retrieved_docs[:3] if retrieved_docs else [],  # 只保存前3条
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
                
                # TensorBoard记录
                log_to_tensorboard(tb_writer, step=idx+1, metrics={
                    "accuracy": accuracy2,
                    "recall_score": recall2,
                    "quality_score": quality_metrics2["quality_score"],
                    "answer_length": len(model_answer2),
                    "similarity": quality_metrics2["similarity"],
                    "retrieved_count": len(retrieved_docs)
                }, scenario="scenario2_with_prompt")
                
                test_results_scenario2.append(result2)
                
            except Exception as e:
                print(f"❌ 场景2测试失败：{e}")
                test_results_scenario2.append({
                    "scenario": "with_prompt_rag",
                    "test_case_id": idx + 1,
                    "question": question,
                    "error": str(e)
                })
        else:
            print("❌ 场景2跳过：FAISS索引未构建")
            test_results_scenario2.append({
                "scenario": "with_prompt_rag",
                "test_case_id": idx + 1,
                "question": question,
                "error": "FAISS索引未构建"
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
        avg_retrieved = sum([r.get("retrieved_count", 0) for r in valid_scenario2]) / len(valid_scenario2)
        
        print(f"\n📊 场景2统计（有提示词RAG）：")
        print(f"  平均准确率：{total_accuracy2:.4f}")
        print(f"  平均召回率：{avg_recall2:.4f}")
        print(f"  平均质量分：{avg_quality2:.4f}")
        print(f"  平均回答长度：{avg_length2:.1f}")
        print(f"  平均检索文档数：{avg_retrieved:.1f}")
        print(f"  有效测试用例：{len(valid_scenario2)}/{len(test_cases)}")
    
    # 对比分析
    if test_results_scenario1 and test_results_scenario2:
        print(f"\n📊 场景对比分析：")
        
        # 计算准确率提升
        if total_accuracy1 > 0 and total_accuracy2 > 0:
            accuracy_improvement = ((total_accuracy2 - total_accuracy1) / total_accuracy1) * 100
            print(f"  准确率提升：{accuracy_improvement:+.2f}%")
        
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
                "avg_retrieved": avg_retrieved if 'avg_retrieved' in locals() else 0,
                "valid_cases": len(valid_scenario2) if 'valid_scenario2' in locals() else 0
            },
            "improvements": {
                "accuracy_improvement": accuracy_improvement if 'accuracy_improvement' in locals() else 0,
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
    print("场景2：有提示词RAG生成（测试提示词优化效果）")
    print("="*60)
    
    run_dual_scenario_test()