import axios from 'axios'

// 根据环境变量设置基础URL（Vite 使用 import.meta.env）
const baseURL = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api'

const request = axios.create({
  baseURL,
  timeout: 60000
})

export async function askQuestion(query, useRerank = false, maxNewTokens = 500) {
  const response = await request.post('/v1/ask', {
    query,
    use_rerank: useRerank,
    max_new_tokens: maxNewTokens
  })
  return response.data
}