<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <div class="logo">招</div>
      <div>
        <h2>招投标智能助手</h2>
        <p>垂类RAG · 客服系统</p>
      </div>
    </div>

    <!-- 快捷问答卡片 -->
    <div class="quick-questions">
      <h4>⚡ 快捷问答</h4>
      <button @click="$emit('quick-question', '上海仓祥绿化工程有限公司的注册地址')">
        上海仓祥的注册地址
      </button>
      <button @click="$emit('quick-question', '恒安达消防设备的价格')">
        恒安达消防设备的价格
      </button>
      <button @click="$emit('quick-question', '2026年三校区校园保安服务的采购方')">
        三校区保安服务采购方
      </button>
    </div>

    <!-- 知识库状态卡片 -->
    <div class="info-card">
      <h4>📊 知识库状态</h4>
      <div class="stat-item"><span>文档总数</span> <strong>{{ stats.totalDocs }}</strong></div>
      <div class="stat-item"><span>最后更新</span> <strong>{{ stats.lastBuild }}</strong></div>
      <div class="stat-item"><span>向量维度</span> <strong>1024</strong></div>
      <div class="stat-item"><span>检索模式</span> <strong>混合搜索</strong></div>
    </div>

    <!-- 系统状态卡片 -->
    <div class="system-status">
      <h4>🖥️ 系统状态</h4>
      <div class="status-item">
        <span class="status-dot"></span> 后端服务：正常
      </div>
      <div class="status-item">
        <span class="status-dot"></span> 模型：Qwen2.5-3B (4bit)
      </div>
      <div class="status-item">
        <span class="status-dot" :class="{ 'warning': !rerankEnabled }"></span>
        重排序：{{ rerankEnabled ? '已启用' : '未启用' }}
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
defineEmits(['clear', 'manage', 'quick-question'])
</script>

<style scoped>
.sidebar {
  width: 300px;
  background: white;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  padding: 28px 20px;
  overflow-y: auto;          /* 允许垂直滚动 */
  max-height: 100%;          /* 确保高度限制 */
  scrollbar-width: thin;      /* Firefox 滚动条宽度 */
  scrollbar-color: #fb7299 #f0f0f0; /* Firefox 滚动条颜色 */
}

/* WebKit 滚动条样式 */
.sidebar::-webkit-scrollbar {
  width: 6px;
}
.sidebar::-webkit-scrollbar-track {
  background: #f0f0f0;
  border-radius: 10px;
}
.sidebar::-webkit-scrollbar-thumb {
  background: #fb7299;
  border-radius: 10px;
}
.sidebar::-webkit-scrollbar-thumb:hover {
  background: #f25d8e;
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}
.logo {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, #fb7299, #fc8bab);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 22px;
  box-shadow: 0 8px 16px -4px rgba(251, 114, 153, 0.2);
}
.sidebar-header h2 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #fb7299;
  margin: 0;
}
.sidebar-header p {
  font-size: 0.75rem;
  color: #9ca3af;
  margin-top: 2px;
}
.quick-questions,
.info-card,
.system-status {
  background: white;
  border-radius: 20px;
  padding: 18px;
  margin-bottom: 20px;
  border: 1px solid #e5e7eb;
}
.quick-questions h4,
.info-card h4,
.system-status h4 {
  font-size: 0.9rem;
  color: #fb7299;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.quick-questions button {
  display: block;
  width: 100%;
  background: #f9f9f9;
  border: 1px solid #e5e7eb;
  border-radius: 40px;
  padding: 10px 12px;
  margin-bottom: 8px;
  font-size: 0.85rem;
  color: #fb7299;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.quick-questions button:hover {
  background: #ffe6ec;
  border-color: #fb7299;
}
.stat-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  margin-bottom: 12px;
  color: #1e293b;
}
.stat-item span:first-child {
  color: #6b7280;
}
.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 0.85rem;
}
.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #10b981;
  display: inline-block;
}
.status-dot.warning {
  background: #f59e0b;
}
.action-buttons {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.action-btn {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 40px;
  padding: 12px;
  font-size: 0.9rem;
  font-weight: 500;
  color: #fb7299;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}
.action-btn:hover {
  background: #f9f9f9;
  border-color: #fb7299;
}
.action-btn.primary {
  background: #fb7299;
  border-color: #fb7299;
  color: white;
}
.action-btn.primary:hover {
  background: #f25d8e;
}
</style>