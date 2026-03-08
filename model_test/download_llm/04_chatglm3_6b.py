# 仅下载不测试的脚本（base 环境已运行完成，模型已存入 /mnt/workspace/data/modelscope/cache）
from modelscope.hub.snapshot_download import snapshot_download
import os
import shutil

model_name = "ZhipuAI/chatglm3-6b"
cache_dir = "/root/.cache/modelscope"
model_cache_path = os.path.join(cache_dir, "ZhipuAI", "chatglm3-6b")


# 下载模型（仅 base 环境运行一次，已完成）
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir=cache_dir
)
print(f"模型下载完成，本地路径：{model_dir}")