import torch
import json
import numpy as np
import faiss
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from torch.utils.tensorboard import SummaryWriter
import sys

# ---------------------- 1. 新增：初始化output目录（统一存储所有文件） ----------------------
def init_output_dir():
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 按时间戳创建TensorBoard日志目录（避免覆盖）
    tb_log_dir = os.path.join(output_dir, f"tb_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(tb_log_dir, exist_ok=True)
    # 测试结果JSON目录
    result_dir = os.path.join(output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)
    # 运行日志目录
    log_dir = os.path.join(output_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    return output_dir, tb_log_dir, result_dir, log_dir

# ---------------------- 2. 新增：日志重定向（终端+文件双输出） ----------------------
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

# ---------------------- 3. 新增：TensorBoard初始化与指标记录 ----------------------
def init_tensorboard(tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"✅ TensorBoard日志已初始化，存储路径：{tb_log_dir}")
    return writer

def log_to_tensorboard(writer, step, metrics):
    """记录核心指标到TensorBoard：召回率、回答长度、准确率（可选）"""
    writer.add_scalar("召回率/单条用例", metrics["recall_score"], step)
    writer.add_scalar("回答长度/模型生成", metrics["answer_length"], step)
    writer.add_scalar("回答长度/标准答案", metrics["reference_length"], step)
    if "accuracy" in metrics:
        writer.add_scalar("准确率/单条用例", metrics["accuracy"], step)

# ---------------------- 4. 配置量化参数（适配轻量级大模型，节省DSW显存） ----------------------
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

# ---------------------- 5. 加载本地模型（适配阿里云DSW，绝对路径+强制本地加载） ----------------------
def load_local_models(llm_name, llm_local_path, embedding_local_path):
    print(f"正在加载本地大模型：{llm_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_local_path,
            trust_remote_code=True,
            padding_side="right",
            local_files_only=True
        )
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
    
    print(f"正在加载本地Embedding模型：{embedding_local_path.split('/')[-1]}")
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
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        embedding_model.eval()
        print(f"✅ Embedding模型 加载成功")
    except Exception as e:
        print(f"❌ Embedding模型加载失败，错误信息：{e}")
        return None, None, None
    
    return tokenizer, llm_model, (embedding_tokenizer, embedding_model)

# ---------------------- 6. 封装Qwen3-Embedding-8B 文本转向量函数 ----------------------
def qwen3_embedding_encode(embedding_models, text):
    embedding_tokenizer, embedding_model = embedding_models
    text = str(text).strip()
    if not text:
        return np.array([])
    
    inputs = embedding_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to("cuda")
    
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    
    hidden_state = outputs.last_hidden_state.to(dtype=torch.float32, device="cpu")
    vec = hidden_state.mean(dim=1).squeeze().numpy()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ---------------------- 7. 构建本地FAISS向量索引 ----------------------
def build_local_vector_index(embedding_models, docs):
    doc_vectors = []
    valid_docs = []
    for doc in docs:
        vec = qwen3_embedding_encode(embedding_models, doc)
        if vec.size > 0:
            doc_vectors.append(vec)
            valid_docs.append(doc)
    
    if not doc_vectors:
        print("❌ 无有效文档向量，无法构建FAISS索引")
        return None, None
    
    doc_vectors = np.array(doc_vectors, dtype=np.float32)
    vector_dim = doc_vectors.shape[1]
    index = faiss.IndexFlatIP(vector_dim)
    index.add(doc_vectors)
    
    print(f"✅ 共构建 {len(valid_docs)} 条有效文档的FAISS索引")
    return index, doc_vectors

# ---------------------- 8. 完整RAG流程 ----------------------
def local_rag_inference(tokenizer, llm_model, embedding_models, index, docs, question):
    q_vec = qwen3_embedding_encode(embedding_models, question)
    if q_vec.size == 0:
        return "问题内容无效，无法生成回答", []
    
    distance, idx = index.search(q_vec.reshape(1, -1), k=2)
    retrieved_docs = [docs[i] for i in idx[0]]
    
    context = "\n".join(retrieved_docs)
    prompt = f"""请基于以下检索到的文档内容，准确回答问题，回答简洁明了，无需额外冗余信息。
文档内容：{context}
问题：{question}
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
            top_p=0.8,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    model_answer = full_response.replace(prompt, "").strip()
    return model_answer, retrieved_docs

# ---------------------- 9. 加载问答对 ----------------------
def load_project_qa_data(qa_file_path="qa_data/1000_qa_pairs.json"):
    try:
        with open(qa_file_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        test_cases = []
        knowledge_docs = []
        for idx, item in enumerate(qa_data):
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            
            if not question or not answer:
                print(f"⚠️  跳过第 {idx+1} 条数据：question/answer为空")
                continue
            
            reference_answer = answer
            context = answer
            if context not in knowledge_docs:
                knowledge_docs.append(context)
            relevant_doc_idx = knowledge_docs.index(context)
            
            test_cases.append({
                "question": question,
                "reference_answer": reference_answer,
                "relevant_doc_idx": relevant_doc_idx
            })
        
        print(f"✅ 加载 {len(test_cases)} 条有效测试用例，{len(knowledge_docs)} 条知识库文档")
        return test_cases, knowledge_docs
    
    except FileNotFoundError:
        print(f"❌ 未找到问答对文件：{qa_file_path}")
        return [], []
    except Exception as e:
        print(f"❌ 加载问答对失败：{e}")
        return [], []

# ---------------------- 10. 核心测试流程（整合所有新增功能） ----------------------
def run_lightweight_selection_test():
    # 初始化目录（output及子目录）
    output_dir, tb_log_dir, result_dir, log_dir = init_output_dir()
    # 初始化日志重定向（保存到output/run_logs）
    log_file_path = os.path.join(log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = LoggerRedirect(log_file_path)
    # 初始化TensorBoard
    tb_writer = init_tensorboard(tb_log_dir)
    
    # 模型配置
    test_config = {
        "llm_name": "Qwen2.5-7B-Instruct",
        "llm_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-7B-Instruct",
        "embedding_local_path": "/mnt/workspace/data/modelscope/cache/qwen/Qwen3-Embedding-8B"
    }
    
    # 1. 加载数据
    qa_file_path = "qa_data/1000_qa_pairs.json"
    test_cases, test_docs = load_project_qa_data(qa_file_path)
    if not test_cases or not test_docs:
        print("\n❌ 无有效测试数据，测试终止")
        tb_writer.close()
        return
    
    # 2. 加载模型
    tokenizer, llm_model, embedding_models = load_local_models(
        test_config["llm_name"],
        test_config["llm_local_path"],
        test_config["embedding_local_path"]
    )
    if tokenizer is None or llm_model is None or embedding_models is None:
        print("\n❌ 模型加载失败，测试终止")
        tb_writer.close()
        return
    
    # 3. 构建FAISS索引
    print("\n正在构建FAISS向量索引...")
    index, _ = build_local_vector_index(embedding_models, test_docs)
    if index is None:
        print("\n❌ FAISS索引构建失败，测试终止")
        tb_writer.close()
        return
    print("✅ FAISS索引构建完成")
    
    # 4. 执行测试
    test_results = []
    print("\n=====================================")
    print("开始轻量级模型选型测试...")
    print("=====================================")
    
    for idx, case in enumerate(test_cases):
        question = case["question"]
        reference_answer = case["reference_answer"]
        relevant_doc_idx = case["relevant_doc_idx"]
        
        # RAG推理
        model_answer, retrieved_docs = local_rag_inference(
            tokenizer, llm_model, embedding_models, index, test_docs, question
        )
        
        # 计算核心指标
        relevant_doc = test_docs[relevant_doc_idx]
        recall_score = 1 if relevant_doc in retrieved_docs else 0
        answer_length = len(model_answer)
        reference_length = len(reference_answer)
        # 可选：手动标注准确率（1=准确，0=不准确），这里默认1，可根据实际需求修改
        accuracy = 1 if reference_answer in model_answer or model_answer in reference_answer else 0
        
        # 记录结果
        single_result = {
            "test_case_id": idx + 1,
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "retrieved_docs": retrieved_docs,
            "relevant_doc": relevant_doc,
            "recall_score": recall_score,
            "accuracy": accuracy,
            "answer_length": answer_length,
            "reference_length": reference_length
        }
        
        # 实时打印
        print(f"\n--- 测试用例 {idx+1} 完成 ---")
        print(f"问题：{question}")
        print(f"模型回答：{model_answer}")
        print(f"召回率：{recall_score} | 准确率：{accuracy} | 回答长度：{answer_length}")
        
        # 写入TensorBoard（step=用例ID）
        log_to_tensorboard(tb_writer, step=idx+1, metrics={
            "recall_score": recall_score,
            "accuracy": accuracy,
            "answer_length": answer_length,
            "reference_length": reference_length
        })
        
        test_results.append(single_result)
    
    # 5. 保存测试结果到output/test_results
    result_file_name = f"{test_config['llm_name']}_dsw_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_file_path = os.path.join(result_dir, result_file_name)
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    # 6. 测试总结
    print("\n=====================================")
    print(f"✅ 测试完成！所有文件已保存到 output 目录")
    print(f"📊 测试结果JSON：{result_file_path}")
    print(f"📈 TensorBoard日志：{tb_log_dir}")
    print(f"📝 运行日志：{log_file_path}")
    total_recall = sum([res["recall_score"] for res in test_results]) / len(test_results)
    total_accuracy = sum([res["accuracy"] for res in test_results]) / len(test_results)
    print(f"📊 平均召回率：{total_recall:.2f} | 平均准确率：{total_accuracy:.2f}")
    print("=====================================")
    
    # 关闭TensorBoard写入器
    tb_writer.close()

# ---------------------- 11. 一键运行 ----------------------
if __name__ == "__main__":
    run_lightweight_selection_test()