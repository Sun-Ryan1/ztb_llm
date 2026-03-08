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
  role: { type: String, required: true }, // 'user' or 'assistant'
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
/* 复制原 index.html 中对应的样式 */
.message { margin-bottom: 24px; max-width: 85%; clear: both; }
.message.user { float: right; }
.message.assistant { float: left; }
.bubble { padding: 14px 18px; border-radius: 20px; line-height: 1.5; font-size: 0.95rem; }
.message.user .bubble { background: #2a5298; color: white; border-bottom-right-radius: 4px; }
.message.assistant .bubble { background: white; color: #1e2b3c; border-bottom-left-radius: 4px; border: 1px solid #e9eef2; }
.meta { font-size: 0.7rem; color: #8a9bb5; margin-top: 6px; display: flex; gap: 12px; }
.message.user .meta { justify-content: flex-end; }
.references { margin-top: 12px; background: #f0f4fa; border-radius: 12px; padding: 10px 14px; }
.doc-item { margin-top: 12px; padding: 10px; background: white; border-radius: 8px; border-left: 3px solid #2a5298; }
.doc-similarity { color: #e67e22; font-weight: 600; font-size: 0.75rem; }
/* 其他样式可继续补充 */
</style>