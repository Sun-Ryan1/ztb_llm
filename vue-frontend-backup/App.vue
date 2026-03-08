<template>
  <div class="app-container">
    <Sidebar @clear="clearConversation" @manage="openManage" :rerankEnabled="useRerank" />
    <ChatView ref="chatView" v-model:useRerank="useRerank" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import ChatView from './views/ChatView.vue'

const useRerank = ref(false)
const chatView = ref(null)

const clearConversation = () => {
  if (chatView.value) {
    chatView.value.clearMessages()
  }
}
const openManage = () => {
  alert('系统管理功能开发中...')
}
</script>

<style>
.app-container {
  width: 1400px;
  max-width: 95vw;
  height: 95vh;
  background: white;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.08);
  display: flex;
  overflow: hidden;
  margin: auto;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f0f2f5;
  margin: 0;
  height: 100vh;
  position: relative;
}
</style>