<template>
  <div class="app-container">
    <Sidebar 
      @clear="clearConversation" 
      @manage="openManage" 
      @quick-question="handleQuickQuestion"
      :rerankEnabled="useRerank" 
    />
    <ChatView ref="chatView" v-model:useRerank="useRerank" v-model:useLangChain="useLangChain" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './views/ChatView.vue'

const useRerank = ref(false)
const useLangChain = ref(false)
const chatView = ref(null)

const clearConversation = () => {
  if (chatView.value) {
    chatView.value.clearMessages()
  }
}
const openManage = () => {
  alert('系统管理功能开发中...')
}
const handleQuickQuestion = (q) => {
  if (chatView.value) {
    chatView.value.setQuestion(q)
    chatView.value.sendQuestion()
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #fef9e7;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0;
  color: #1e293b;
}

.app-container {
  width: 100%;
  max-width: 1400px;
  height: 95vh;
  background: white;
  border-radius: 32px;
  box-shadow: 0 25px 50px -12px rgba(249, 115, 22, 0.15);
  display: flex;
  overflow: hidden;
  margin: 0 20px;
}
</style>