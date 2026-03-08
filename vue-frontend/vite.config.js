import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    host: '0.0.0.0',
    allowedHosts: [
      '685986-proxy-5173.dsw-gateway-cn-shanghai.data.aliyun.com',
      '.dsw-gateway-cn-shanghai.data.aliyun.com' // 允许所有该域子域名
    ],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})