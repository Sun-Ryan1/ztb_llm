# download_bge_model.py
import os
from modelscope import snapshot_download

# 模型名称
model_name = "BAAI/bge-large-zh-v1.5"

# 本地保存路径（推荐）
local_save_path = "/mnt/workspace/data/modelscope/cache/bge-large-zh-v1.5"

print(f"正在下载模型: {model_name}")
print(f"保存到: {local_save_path}")

try:
    # 下载模型
    model_path = snapshot_download(
        model_name,
        cache_dir=local_save_path,
        local_files_only=False  # 如果本地没有，则从远程下载
    )
    
    print(f"✅ 模型下载完成!")
    print(f"模型路径: {model_path}")
    
    # 验证下载
    files = os.listdir(model_path)
    print(f"模型包含文件: {len(files)} 个")
    for file in sorted(files)[:10]:  # 显示前10个文件
        print(f"  - {file}")
        
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("\n备用方案：使用HuggingFace下载")
    
    # 备用方案
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    try:
        print("尝试通过transformers下载...")
        
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 下载模型
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 保存到本地
        save_path = local_save_path
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"✅ 通过transformers下载并保存完成!")
        print(f"保存路径: {save_path}")
        
    except Exception as e2:
        print(f"❌ 所有下载方式都失败: {e2}")