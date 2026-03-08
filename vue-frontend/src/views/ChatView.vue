<template>
  <div class="chat-main">
    <div class="chat-header">
      <h3>智能问答 · 客服机器人</h3>
      <div class="status"><span class="status-dot"></span> 服务正常</div>
    </div>

    <div class="chat-messages" ref="messagesContainer">
      <div v-for="(msg, idx) in messages" :key="idx">
        <ChatMessage
          :role="msg.role"
          :content="msg.content"
          :references="msg.references"
          :timestamp="msg.timestamp"
          @feedback="handleFeedback"
        />
        <!-- 只在最新一条助手消息下方显示相关推荐 -->
        <div v-if="msg.role === 'assistant' && idx === lastAssistantIndex" class="related-links">
          <span>相关推荐：</span>
          <a @click="quickQuestion('该产品的规格是什么')">规格</a>
          <a @click="quickQuestion('还有哪些供应商')">其他供应商</a>
          <a @click="quickQuestion('历史中标项目')">中标项目</a>
        </div>
      </div>
    </div>

    <!-- 优化后的输入区域：复选框在上，输入框在下 -->
    <div class="chat-input-area">
      <div class="checkbox-group">
        <div class="rerank-checkbox">
          <input
            type="checkbox"
            id="rerankCheckbox"
            :checked="useRerank"
            @change="$emit('update:useRerank', $event.target.checked)"
          />
          <label for="rerankCheckbox">启用重排序</label>
        </div>
        <div class="langchain-checkbox">
          <input
            type="checkbox"
            id="langchainCheckbox"
            :checked="useLangChain"
            @change="$emit('update:useLangChain', $event.target.checked)"
          />
          <label for="langchainCheckbox">使用 LangChain</label>
        </div>
      </div>
      <div class="input-wrapper">
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
    </div>

    <!-- 底部反馈链接 -->
    <div class="footer-note">
      <span>如果存在不符合事实的信息，请提供反馈，帮助我们改进！</span>
      <a @click="openFeedback">📝 发送反馈</a>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, computed } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import { askQuestion } from '@/api'

const props = defineProps({
  useRerank: Boolean,
  useLangChain: Boolean
})
const emit = defineEmits(['update:useRerank', 'update:useLangChain'])

const messages = ref([
  { role: 'assistant', content: '您好！我是招投标智能客服，您可以问我关于公司信息、产品价格、中标项目、法规条款等问题。例如：“上海仓祥绿化工程有限公司的注册地址”', timestamp: new Date() }
])
const question = ref('')
const loading = ref(false)
const messagesContainer = ref(null)

// 计算最后一条助手消息的索引
const lastAssistantIndex = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    if (messages.value[i].role === 'assistant') {
      return i
    }
  }
  return -1
})

// 快捷问题点击
const quickQuestion = (q) => {
  question.value = q
  sendQuestion()
}

// 滚动到底部（平滑滚动）
const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: messagesContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

// 发送问题
const sendQuestion = async () => {
  if (!question.value.trim()) return

  messages.value.push({ role: 'user', content: question.value, timestamp: new Date() })
  await scrollToBottom()

  const q = question.value
  question.value = ''
  loading.value = true

  try {
    // 根据是否使用 LangChain 选择不同的 API 端点
    const endpoint = props.useLangChain ? '/v1/ask_direct_langchain' : '/v1/ask';
    const data = await askQuestion(q, props.useRerank, endpoint)
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      references: data.retrieved_documents,
      timestamp: new Date()
    })
  } catch (error) {
    console.error('请求失败:', error)
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

// 消息反馈（赞/踩）
const handleFeedback = (type) => {
  alert(`感谢您的${type === 'good' ? '好评' : '反馈'}！我们会持续优化。`)
}

// 打开反馈
const openFeedback = () => {
  alert('反馈功能开发中，感谢您的建议！')
}

// 暴露方法给父组件
defineExpose({
  setQuestion: (q) => { question.value = q },
  sendQuestion,
  clearMessages: () => {
    messages.value = [
      { role: 'assistant', content: '您好！我是招投标智能客服，您可以问我关于公司信息、产品价格、中标项目、法规条款等问题。例如：“上海仓祥绿化工程有限公司的注册地址”', timestamp: new Date() }
    ]
  }
})
</script>

<style scoped>
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
  min-width: 0;
}
.chat-header {
  padding: 20px 24px;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}
.chat-header h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: #fb7299;
  margin: 0;
}
.status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  color: #10b981;
  white-space: nowrap;
}
.status-dot {
  width: 8px;
  height: 8px;
  background: #10b981;
  border-radius: 50%;
  display: inline-block;
}
.chat-messages {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  background: #f9fafc;
  scroll-behavior: smooth;
}
.related-links {
  margin-top: 8px;
  margin-bottom: 16px;
  font-size: 0.8rem;
  color: #00a1d6;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.related-links a {
  color: #00a1d6;
  text-decoration: underline;
  cursor: pointer;
  opacity: 0.8;
}
.related-links a:hover {
  opacity: 1;
}
/* 输入区域整体 */
.chat-input-area {
  padding: 16px 24px;
  background: white;
  border-top: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
/* 复选框组：水平排列，间距小，不换行 */
.checkbox-group {
  display: flex;
  gap: 20px;                /* 两个复选框之间的间距 */
  align-items: center;
  flex-wrap: wrap;           /* 如果空间不足，允许换行 */
}
.rerank-checkbox,
.langchain-checkbox {
  display: flex;
  align-items: center;
  gap: 6px;                 /* 复选框与标签间距 */
  font-size: 0.9rem;        /* 适当增大字体，提高可读性 */
  white-space: nowrap;
}
.rerank-checkbox {
  color: #f97316;
}
.rerank-checkbox input {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: #f97316;
}
.langchain-checkbox {
  color: #00a1d6;
}
.langchain-checkbox input {
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: #00a1d6;
}
/* 输入框和按钮 */
.input-wrapper {
  display: flex;
  gap: 12px;
  align-items: center;
  width: 100%;
}
.input-wrapper input {
  flex: 1;
  min-width: 200px;
  padding: 14px 20px;
  border: 2px solid #ffedd5;
  border-radius: 50px;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  background: #fffaf0;
}
.input-wrapper input:focus {
  border-color: #f97316;
  box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.1);
  background: white;
}
.input-wrapper button {
  background: #f97316;
  color: white;
  border: none;
  border-radius: 50px;
  padding: 14px 30px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 8px 16px -4px rgba(249, 115, 22, 0.2);
  white-space: nowrap;
}
.input-wrapper button:hover:not(:disabled) {
  background: #ea580c;
  transform: translateY(-2px);
  box-shadow: 0 12px 20px -6px rgba(249, 115, 22, 0.3);
}
.input-wrapper button:disabled {
  background: #fdba74;
  box-shadow: none;
  cursor: not-allowed;
}
.loading-spinner {
  display: inline-block;
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255,255,255,0.3);
  border-radius: 50%;
  border-top-color: white;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.footer-note {
  text-align: center;
  font-size: 0.7rem;
  color: #94a3b8;
  padding: 12px;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: center;
  gap: 24px;
  flex-wrap: wrap;
}
.footer-note a {
  color: #fb7299;
  text-decoration: none;
  cursor: pointer;
}
.footer-note a:hover {
  text-decoration: underline;
}
</style>