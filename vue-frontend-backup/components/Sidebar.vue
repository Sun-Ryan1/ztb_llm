<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <div class="logo">招</div>
      <div>
        <h2>招投标智能助手</h2>
        <p>垂类RAG · 客服系统</p>
      </div>
    </div>

    <div class="info-card">
      <h4>📊 知识库状态</h4>
      <div class="stat-item"><span>文档总数</span> <strong>{{ stats.totalDocs }}</strong></div>
      <div class="stat-item"><span>最后更新</span> <strong>{{ stats.lastBuild }}</strong></div>
      <div class="stat-item"><span>向量维度</span> <strong>1024</strong></div>
      <div class="stat-item"><span>检索模式</span> <strong>混合搜索</strong></div>
    </div>

    <div class="info-card">
      <h4>🤖 模型信息</h4>
      <div class="model-tag">LLM: Qwen2.5-3B-Instruct (4bit)</div>
      <div class="model-tag">Embedding: BGE-M3</div>
      <div class="model-tag" :class="{ 'rerank-enabled': rerankEnabled }">
        Rerank: {{ rerankEnabled ? '已启用' : '未启用' }}
      </div>
    </div>

    <div class="action-buttons">
      <button class="action-btn" @click="$emit('clear')">🗑️ 清空对话</button>
      <button class="action-btn primary" @click="$emit('manage')">⚙️ 系统管理</button>
    </div>
  </div>
</template>

<script setup>
defineProps({
  stats: { type: Object, default: () => ({ totalDocs: 187091, lastBuild: '2026-02-14' }) },
  rerankEnabled: { type: Boolean, default: false }
})
defineEmits(['clear', 'manage'])
</script>

<style scoped>
.sidebar { width: 280px; background: #f8fafd; border-right: 1px solid #e9eef2; padding: 24px 16px; }
.sidebar-header { display: flex; align-items: center; gap: 8px; margin-bottom: 32px; }
.logo { width: 40px; height: 40px; background: linear-gradient(135deg, #1e3c72, #2a5298); border-radius: 12px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 20px; }
.info-card { background: white; border-radius: 16px; padding: 16px; margin-bottom: 24px; border: 1px solid #e6eaf0; }
.stat-item { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 10px; }
.model-tag { background: #f0f3f8; padding: 6px 12px; border-radius: 30px; font-size: 0.8rem; margin-bottom: 8px; display: inline-block; }
.action-buttons { margin-top: auto; display: flex; flex-direction: column; gap: 8px; }
.action-btn { background: white; border: 1px solid #d0d9e8; border-radius: 40px; padding: 12px; font-size: 0.9rem; cursor: pointer; text-align: center; }
.action-btn.primary { background: #2a5298; color: white; }
</style>