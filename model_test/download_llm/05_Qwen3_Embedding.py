from modelscope.hub.snapshot_download import snapshot_download

# 下载Qwen3-Embedding-8B（ModelScope官方版本）
model_local_dir = snapshot_download(
    model_id="qwen/Qwen3-Embedding-8B",
    cache_dir="/mnt/workspace/data/modelscope/cache"
)
print(f"Qwen3-Embedding-8B下载完成，路径：{model_local_dir}")