from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ModelScope上InternLM2.5-7B-Chat的正确ID
model_name = "Shanghai_AI_Laboratory/internlm2_5-7b-chat"

# 1. 从ModelScope下载模型（保留原始参数，无修改）
print("开始下载模型...")
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir="/mnt/workspace/data/modelscope/cache"
)
print(f"模型下载完成，本地路径：{model_dir}")

# 2. 定义高版本兼容的4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=False
)

# 3. 加载Tokenizer和模型（修正 torch_dtype 为 dtype，解决弃用警告）
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16,  # 核心修正：替换弃用的 torch_dtype 为 dtype
    attn_implementation="eager"
).eval()

# 4. 测试基础功能（移除重复的 attention_mask 参数，解决 TypeError）
prompt = "你是InternLM2.5大模型，请介绍一下你自己"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=False,
    truncation=True,
    max_length=512
).to("cuda" if torch.cuda.is_available() else "cpu")

# 修正核心：删除显式的 attention_mask 参数，避免重复传入
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    use_cache=False  # 保留关闭缓存，解决维度冲突的核心优化
)

print("模型测试输出：")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))