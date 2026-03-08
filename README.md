# 基于大模型的数字招投标采购领域应用

## 1. 项目简介 🚀

本项目旨在构建一个面向招投标采购全流程的智能问答系统，集成政策法规、招标信息、中标信息、企业信息、商品信息及价格数据六大核心模块，为采购人、供应商、监管机构等用户提供精准、高效的信息检索与问答服务。

- 采用 **RAG 架构**，结合向量检索与大语言模型，实现精准问答
- 支持 **多种主流中文 LLM**（Qwen、GLM、LLaMA、IntelLM）
- 集成 **向量数据库** 与 **混合检索**，提升检索准确率
- 后端基于 **FastAPI** 构建，前端基于 **Vue 3 + Vite** 开发

### 关键成果概览

| 指标 | 数值 |
|------|------|
| 知识库规模 | 18.7 万份知识文档，覆盖六大领域 |
| 问答准确率 | RAG 问答准确率达 **76%** |
| 系统性能 | 平均检索 0.3s，生成 1.2s，总响应 1.5s |
| 技术栈 | Qwen2.5-3B + bge-m3 + ChromaDB |

## 2. 项目预览 📸

### 前端界面

![前端界面](assets/system_frontend.png)



## 3. 项目特性 ✨

- 🔍 **智能检索**：混合检索（向量 + BM25）+ Cross-Encoder 重排序
- 🤖 **多模型支持**：可插拔接入 Qwen、GLM、LLaMA、IntelLM 等多种大模型
- 📚 **RAG 知识库**：基于 ChromaDB 的向量数据库，支持 18 万+ 文档
- 🌐 **前后端分离**：FastAPI 后端 + Vue 3 前端
- 📱 **现代化 UI**：响应式设计，实时聊天界面
- 🚀 **检索优化**：智能查询分类、地名扩展、线程安全缓存
- 🧩 **灵活配置**：支持环境变量和 YAML 配置文件

## 4. 为什么有用 💡

- **专业领域问答**：专注于招投标领域，提供公司、产品、法规等专业问答
- **高准确率**：混合检索 + 重排序，显著提升检索准确率
- **多场景应用**：可用于招标信息查询、公司资质查询、法规咨询等场景
- **可扩展架构**：支持自定义模型和检索策略扩展
- **开源免费**：完全开源，可自由定制和二次开发

## 5. 支持的查询类型 📋

| 查询类型 | 说明 | 示例 |
|---------|------|------|
| `credit_code` | 统一社会信用代码查询 | 查询某公司的信用代码 |
| `company_name` | 公司名称查询 | 根据名称查询公司信息 |
| `address` | 地址查询 | 查询某地区的公司 |
| `legal_representative` | 法定代表人查询 | 查询某法人名下公司 |
| `product_keyword` | 产品关键词查询 | 搜索相关产品 |
| `zhaobiao_natural` | 招投标自然语言查询 | 查询招标项目信息 |
| `law_natural` | 法律法规查询 | 查询相关法规条款 |
| `cross_city` | 跨城市查询 | 多地区联合查询 |

## 6. 技术架构 🏗️

### 6.1 后端核心模块

| 模块 | 文件 | 功能描述 |
|:----:|:----:|:--------:|
| 配置管理 | `config.py` | 定义 AppConfig 数据类，支持从 YAML 文件或环境变量加载配置 |
| 依赖注入 | `dependencies.py` | 提供 FastAPI 依赖函数，通过 lru_cache 获取全局配置 |
| LLM集成 | `llm_integration.py` | 实现 LLMEngine 类，封装模型加载、量化、推理、批处理、缓存等功能 |
| 提示工程 | `prompt_engineering.py` | 提供 PromptManager 管理多种提示模板 |
| 检索增强 | `retrieval.py` | 实现 RetrievalEngine 类，封装向量检索、混合检索、查询分类、重排序等功能 |
| 工具函数 | `utils.py` | 提供公司名称提取、最佳文档选择等工具 |

### 6.2 模块协作流程

1. 用户请求进入 FastAPI 应用，通过依赖注入获取 `LLMEngine`、`RetrievalEngine`、`PromptManager` 实例
2. `RetrievalEngine` 根据用户查询进行分类，执行混合检索（向量+BM25），可选重排序
3. `PromptManager` 根据配置选择专业模板，将检索到的文档片段格式化为上下文
4. `LLMEngine` 接收提示词，调用本地模型生成最终答案，并利用缓存提升响应速度
5. 返回结果前，可调用 `select_best_document` 进一步精炼答案来源

## 7. 快速开始 🚀

### 环境要求

- **Python 3.8+**
- **Node.js 18+**
- **CUDA 11.8+**（可选，用于 GPU 加速）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/Sun-Ryan1/ztb_llm.git
cd ztb_llm
```

2. **安装后端依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

创建 `.env` 文件或设置环境变量：

```bash
# 模型路径
LLM_MODEL_PATH=/path/to/your/llm/model
EMBEDDING_MODEL_PATH=/path/to/your/embedding/model

# 向量数据库路径
CHROMA_DB_PATH=/path/to/chroma/db
```

4. **启动后端服务**

```bash
cd src/app
python main.py
```

后端服务将在 `http://localhost:8000` 启动。

5. **启动前端服务**

```bash
cd vue-frontend
npm install
npm run dev
```

前端服务将在 `http://localhost:5173` 启动。

## 8. 使用示例 📖

### 8.1 API 调用示例

```bash
# 问答接口
curl -X POST "http://localhost:8000/api/v1/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "请查询北京地区的招标项目"}'
```

### 8.2 前端使用

1. 打开浏览器，访问 `http://localhost:5173`
2. 在聊天界面输入问题，例如：
   - "查询北京地区的招标项目"
   - "某公司的统一社会信用代码是多少"
   - "搜索相关的法规条款"
3. 系统将返回相关答案和参考文档

### 8.3 Python 调用示例

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/ask",
    json={"question": "查询北京地区的招标项目"}
)
print(response.json())
```

## 9. 项目结构 📁

```
ztb_llm/
├── src/                           # 源代码
│   ├── app/                       # 主应用
│   │   ├── main.py                # FastAPI 入口
│   │   ├── core_modules/          # 核心模块
│   │   │   ├── config.py          # 配置管理
│   │   │   ├── llm_integration.py # LLM 引擎
│   │   │   ├── retrieval.py       # 检索引擎
│   │   │   └── prompt_engineering.py # 提示工程
│   │   ├── api/                   # API 路由
│   │   ├── adapters/              # LangChain 适配器
│   │   └── data/                  # 数据和知识库
│   ├── crawler_data/              # 爬虫和数据清洗
│   ├── dataset/                   # 数据集
│   └── scripts/                   # 构建脚本
├── vue-frontend/                  # Vue 3 前端
│   ├── src/
│   │   ├── views/                 # 视图组件
│   │   ├── components/            # UI 组件
│   │   ├── stores/                # 状态管理
│   │   └── api/                   # API 调用
│   └── package.json
├── model_test/                    # 模型测试
│   ├── download_llm/              # 模型下载脚本
│   ├── model_select/              # 模型选择测试
│   └── 向量数据库查询/             # 检索测试
├── assets/                        # 项目截图和资源
└── README.md                      # 项目文档
```

## 10. 技术栈 🛠️

### 10.1 后端

| 技术 | 用途 |
|------|------|
| **FastAPI** | Web 框架 |
| **LangChain** | RAG 链构建 |
| **Transformers** | 模型加载和推理 |
| **ChromaDB** | 向量数据库 |
| **Sentence-Transformers** | 文本嵌入 |
| **Jieba** | 中文分词 |
| **BM25 (rank_bm25)** | 关键词检索 |

### 10.2 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| **Vue.js** | 3.5+ | 前端框架 |
| **Vite** | 7.x | 构建工具 |
| **Pinia** | 3.x | 状态管理 |
| **Axios** | 1.x | HTTP 客户端 |

### 10.3 支持的 AI 模型

**LLM 大语言模型：**
- Qwen2.5-3B-Instruct（默认，综合评分最高）
- Qwen2.5-7B（召回率最高，适合复杂问题）
- LLaMA3.2-3B
- ChatGLM3-6B
- IntelLM2-7B

**Embedding 模型：**
- BAAI/bge-m3（默认，召回率 0.9923）
- BAAI/bge-large-zh-v1.5

**Reranker 重排序模型：**
- BAAI/bge-reranker-base
- BAAI/bge-reranker-large

### 10.4 模型选型结论

| 组件 | 推荐方案 | 理由 |
|------|----------|------|
| **大模型** | Qwen2.5-3B-Instruct | 综合评分最高、资源友好、中文能力强 |
| **Embedding模型** | BAAI/bge-m3 | 召回率领先（0.9923）、多语言支持 |
| **向量数据库** | ChromaDB | 部署简单、维护成本低、元数据支持完善 |

## 11. 知识库数据 📊

| 数据类型 | 文档数 | 说明 |
|----------|--------|------|
| company | 162,065 | 公司信息（地址、法人、信用代码等） |
| product | 10,877 | 产品信息 |
| price | 7,292 | 价格信息 |
| zhaobiao | 3,456 | 招标信息 |
| zhongbiao | 3,305 | 中标信息 |
| law | 96 | 法律法规 |

**总计：187,091 条文档**

### 数据处理流程

```
原始数据 → 数据清洗 → 结构化处理 → 知识库构建 → 向量化存储
```

## 12. 开发环境搭建 💻

### 12.1 后端开发

1. **安装 Python 3.8+**
2. **安装 CUDA**（可选，用于 GPU 加速）
3. **安装依赖**：`pip install -r requirements.txt`
4. **下载模型**：运行 `model_test/download_llm/` 下的脚本

### 12.2 前端开发

1. **安装 Node.js 18+**
2. **安装依赖**：`npm install`
3. **启动开发服务器**：`npm run dev`

## 13. API 文档 📝

### 13.1 问答接口

**POST** `/api/v1/ask`

请求体：
```json
{
  "question": "查询北京地区的招标项目",
  "use_rerank": true,
  "top_k": 5
}
```

响应：
```json
{
  "answer": "根据查询结果...",
  "sources": [...],
  "query_type": "zhaobiao_natural"
}
```

### 13.2 健康检查

**GET** `/api/v1/health`

响应：
```json
{
  "status": "healthy"
}
```

## 14. 常见问题 ❓

### Q: 如何添加新的数据源？
A: 在 `src/dataset/unclean_data/` 目录下添加原始数据，然后运行 `src/scripts/build_knowledge_base.py` 构建知识库。

### Q: 如何切换不同的 LLM 模型？
A: 修改 `src/app/core_modules/config.py` 中的 `LLM_MODEL_PATH` 配置，或设置环境变量。

### Q: 如何启用 GPU 加速？
A: 确保安装了 CUDA 和对应的 PyTorch 版本，系统会自动检测并使用 GPU。

### Q: 如何调整检索参数？
A: 在 `src/app/core_modules/retrieval.py` 中修改检索参数，如 `top_k`、`rerank` 等。

## 15. 项目状态 📊

- **开发状态**：活跃开发中
- **最新版本**：1.0.0
- **许可证**：MIT

## 16. 路线图 🗺️

- [x] 基础 RAG 问答功能
- [x] 多模型支持
- [x] 混合检索 + 重排序
- [x] Vue 3 前端界面
- [x] 知识库构建（18.7万文档）
- [ ] 多语言支持
- [ ] 移动端适配
- [ ] 知识库自动更新

## 17. 许可证 📄

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 18. 联系方式 📞

- **项目地址**：https://github.com/Sun-Ryan1/ztb_llm
- **作者**：Sun-Ryan1
- **邮箱**：1819165504@qq.com

## 19. 致谢 🙏

感谢以下开源项目：
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- [Qwen](https://github.com/QwenLM/Qwen)

---

**如果您觉得这个项目有用，请给个 ⭐️ 支持一下！**
