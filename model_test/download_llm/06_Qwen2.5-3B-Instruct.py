from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ModelScope上Qwen2.5-3B-Instruct的正确ID（官方公开版本）
model_name = "qwen/Qwen2.5-3B-Instruct"

# 1. 从ModelScope下载模型（移除不支持的resume_download参数）
print("开始下载模型...")
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir="/mnt/workspace/data/modelscope/cache"  # 仅保留支持的参数
)
print(f"模型下载完成，本地路径：{model_dir}")

# 2. 加载Tokenizer和模型（开启4bit量化，显存占用更低）
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,
    low_cpu_mem_usage=True
).eval()

# 3. 测试基础功能（验证模型可用）
prompt = "你好，请介绍一下自己"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(** inputs, max_new_tokens=100)
print("模型测试输出：")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))