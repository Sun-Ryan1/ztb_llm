from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ModelScope上该模型的ID
model_name = "LLM-Research/Llama-3.2-3B-Instruct"

# 下载模型到ModelScope缓存目录
print("开始下载模型...")
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir="/mnt/workspace/data/modelscope/cache"
)
print(f"模型下载完成，本地路径：{model_dir}")

# 加载模型（3B参数，显存占用更低）
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,
    low_cpu_mem_usage=True
).eval()

# 测试模型
prompt = "请介绍一下你自己"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print("模型输出：")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))