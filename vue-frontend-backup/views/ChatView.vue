<template>
  <div class="chat-main">
    <div class="chat-header">
      <h3>智能问答 · 客服机器人</h3>
      <div class="status"><span class="status-dot"></span> 服务正常</div>
    </div>

    <div class="chat-messages" ref="messagesContainer">
      <ChatMessage
        v-for="(msg, idx) in messages"
        :key="idx"
        :role="msg.role"
        :content="msg.content"
        :references="msg.references"
        :timestamp="msg.timestamp"
        @feedback="handleFeedback"
      />
    </div>

    <div class="chat-input-area">
      <div class="rerank-checkbox">
        <input type="checkbox" id="rerankCheckbox" v-model="useRerank">
        <label for="rerankCheckbox">启用重排序</label>
      </div>
      <input
        type="text"
        v-model="question"
        @keyup.enter="sendQuestion"
        placeholder="请输入您的问题..."
        :disabled="loading"
      />
      <button @click="sendQuestion" :disabled="loading">
        <span v-if="loading" class="loading-spinner"></span>
        <span v-else>发送</span>
      </button>
    </div>

    <div class="footer-note">如果存在不符合事实的信息，请提供反馈，帮助我们改进！</div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import { askQuestion } from '@/api'

const messages = ref([
  { role: 'assistant', content: '您好！我是招投标智能客服...', timestamp: new Date() }
])
const question = ref('')
const loading = ref(false)
const useRerank = ref(false)
const messagesContainer = ref(null)

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const sendQuestion = async () => {
  if (!question.value.trim()) return

  messages.value.push({ role: 'user', content: question.value, timestamp: new Date() })
  await scrollToBottom()

  const q = question.value
  question.value = ''
  loading.value = true

  try {
    const data = await askQuestion(q, useRerank.value)
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      references: data.retrieved_documents,
      timestamp: new Date()
    })
  } catch (error) {
    messages.value.push({
      role: 'assistant',
      content: `抱歉，发生错误：${error.message}。请稍后重试。`,
      timestamp: new Date()
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

const handleFeedback = (type) => {
  alert(`感谢您的${type === 'good' ? '好评' : '反馈'}！我们会持续优化。`)
}
</script>

<style scoped>
.chat-main { flex: 1; display: flex; flex-direction: column; background: white; }
.chat-header { padding: 20px 24px; border-bottom: 1px solid #e9eef2; display: flex; align-items: center; justify-content: space-between; }
.status-dot { width: 8px; height: 8px; background: #3bb55b; border-radius: 50%; display: inline-block; margin-right: 6px; }
.chat-messages { flex: 1; padding: 24px; overflow-y: auto; background: #f9fafc; }
.chat-input-area { padding: 20px 24px; background: white; border-top: 1px solid #e9eef2; display: flex; gap: 12px; align-items: center; }
.chat-input-area input { flex: 1; padding: 14px 20px; border: 1px solid #d0d9e8; border-radius: 40px; font-size: 0.95rem; }
.chat-input-area button { background: #2a5298; color: white; border: none; border-radius: 40px; padding: 14px 32px; cursor: pointer; }
.chat-input-area button:disabled { background: #a0b3d9; }
.loading-spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: white; animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.footer-note { text-align: center; font-size: 0.7rem; color: #8a9bb5; padding: 12px; border-top: 1px solid #e9eef2; }
</style>