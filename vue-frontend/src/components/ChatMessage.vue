<template>
  <div class="message" :class="role">
    <div class="bubble-wrapper">
      <div class="bubble">{{ content }}</div>
      <details v-if="references && references.length" class="references">
        <summary>📚 参考文档 ({{ references.length }})</summary>
        <div v-for="(doc, idx) in references.slice(0,5)" :key="idx" class="doc-item">
          <p><strong>文档 {{ idx+1 }}</strong> 
            <span class="doc-similarity">相似度: {{ doc.similarity?.toFixed(4) }}</span>
          </p>
          <p>{{ doc.content_preview || doc.content.substring(0,150)+'...' }}</p>
        </div>
      </details>
      <div class="meta">
        <span>{{ role === 'user' ? '您' : '助手' }}</span>
        <span>{{ timeStr }}</span>
        <span class="message-actions">
          <button class="action-icon" @click="copyText">📋 复制</button>
          <button class="action-icon" @click="$emit('feedback', 'good')">👍</button>
          <button class="action-icon" @click="$emit('feedback', 'bad')">👎</button>
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, required: true },
  references: { type: Array, default: null },
  timestamp: { type: Date, default: () => new Date() }
})

const emit = defineEmits(['copy', 'feedback'])

const timeStr = computed(() => {
  return props.timestamp.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit' })
})

const copyText = () => {
  navigator.clipboard.writeText(props.content).then(() => {
    alert('已复制到剪贴板')
  }).catch(() => alert('复制失败'))
}
</script>

<style scoped>
.message {
  margin-bottom: 24px;
  max-width: 75%;
  clear: both;
  animation: slideIn 0.2s ease;
}
@keyframes slideIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.message.user {
  float: right;
}
.message.assistant {
  float: left;
}
.bubble {
  padding: 14px 20px;
  border-radius: 22px;
  line-height: 1.6;
  font-size: 0.95rem;
  word-wrap: break-word;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
  position: relative;
}
.message.user .bubble {
  background: #fb7299;
  color: white;
  border-bottom-right-radius: 6px;
}
.message.assistant .bubble {
  background: white;
  color: #1e293b;
  border-bottom-left-radius: 6px;
  border: 1px solid #e5e7eb;
}
.meta {
  font-size: 0.7rem;
  color: #94a3b8;
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.message.user .meta {
  justify-content: flex-end;
}
.message-actions {
  display: flex;
  gap: 4px;
}
.action-icon {
  background: none;
  border: none;
  color: #94a3b8;
  cursor: pointer;
  font-size: 0.8rem;
  padding: 4px 8px;
  border-radius: 16px;
  transition: all 0.2s;
}
.action-icon:hover {
  background: #ffe6ec;
  color: #fb7299;
}
.references {
  margin-top: 16px;
  background: white;
  border-radius: 18px;
  padding: 14px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
}
.references summary {
  cursor: pointer;
  color: #fb7299;
  font-weight: 600;
  font-size: 0.85rem;
  outline: none;
  padding: 4px 0;
}
.doc-item {
  margin-top: 16px;
  padding: 12px;
  background: #f9f9f9;
  border-radius: 14px;
  border-left: 4px solid #fb7299;
  font-size: 0.85rem;
  line-height: 1.5;
}
.doc-item p {
  margin: 4px 0;
}
.doc-similarity {
  color: #f59e0b;
  font-weight: 600;
  font-size: 0.75rem;
  margin-left: 8px;
}
</style>