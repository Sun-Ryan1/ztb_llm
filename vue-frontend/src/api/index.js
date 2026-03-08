import axios from 'axios'

const request = axios.create({
  baseURL: '/api',   // 通过代理转发到后端
  timeout: 60000
})

/**
 * 发送问答请求
 * @param {string} query 用户问题
 * @param {boolean} useRerank 是否启用重排序
 * @param {string} endpoint API 端点，默认为 '/ask'
 * @param {number} maxNewTokens 最大生成 token 数
 * @returns {Promise<Object>} 问答结果
 */
export async function askQuestion(query, useRerank = false, endpoint = '/ask', maxNewTokens = 500) {
  const response = await request.post(endpoint, {
    query,
    use_rerank: useRerank,
    max_new_tokens: maxNewTokens
  })
  return response.data
}