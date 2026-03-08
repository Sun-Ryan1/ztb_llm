import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 从 core_modules 导入所需组件
from core_modules import (
    AppConfig,
    Environment,
    LLMEngine,
    LLMConfig,
    RetrievalEngine,
    RetrievalConfig,
    CrossEncoderReranker,
    PromptManager,
)

# LangChain 相关导入
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from adapters.langchain_retriever import CustomRetriever
from adapters.langchain_llm import CustomLLM

from api import routes

def log_info(msg):
    print(f"[INFO] {msg}")

def log_error(msg):
    print(f"[ERROR] {msg}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config
    try:
        log_info("开始初始化 LLM 引擎...")
        llm_config = LLMConfig(
            model_path=config.llm_model_path,
            load_in_4bit=config.llm_load_in_4bit,
            default_max_new_tokens=config.llm_default_max_new_tokens,
            enable_cache=config.llm_enable_cache,
        )
        app.state.llm_engine = LLMEngine(llm_config)
        log_info("LLM 引擎初始化成功")
    except Exception as e:
        log_error(f"LLM 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        log_info("开始初始化检索引擎...")
        retrieval_config = RetrievalConfig(
            chroma_db_path=config.chroma_db_path,
            embedding_model_path=config.embedding_model_path,
            collection_name=config.collection_name,
        )
        reranker = None
        if config.rerank_enabled:
            reranker = CrossEncoderReranker(
                model_path=config.rerank_model_path
            )
        app.state.retrieval_engine = RetrievalEngine(
            db_path=retrieval_config.chroma_db_path,
            model_path=retrieval_config.embedding_model_path,
            collection_name=retrieval_config.collection_name,
            reranker=reranker,
            location_config_path=config.location_config_path
        )
        log_info("检索引擎初始化成功")
    except Exception as e:
        log_error(f"检索引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        log_info("开始初始化提示管理器...")
        prompt_manager = PromptManager()
        if config.prompt_templates_file:
            prompt_manager.load_from_file(config.prompt_templates_file)
        app.state.prompt_manager = prompt_manager
        log_info("提示管理器初始化成功")
    except Exception as e:
        log_error(f"提示管理器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

   # ---------- 初始化 LangChain 组件 ----------
    log_info("开始初始化 LangChain 适配器...")
    retriever = CustomRetriever(app.state.retrieval_engine)
    llm = CustomLLM(app.state.llm_engine)

    # 简化提示模板，要求直接复制原文，并明确禁止添加前缀
    template_str = """请只输出下面的原文，不要添加任何前缀或解释：

    {context}"""
    prompt_template = PromptTemplate.from_template(template_str)

    def format_docs(docs):
        if docs:
            log_info(f"LangChain 检索到最佳文档: {docs[0].page_content[:100]}...")
            return docs[0].page_content
        else:
            log_info("LangChain 检索结果为空")
            return ""

    # 构建基础链
    base_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.bind(max_new_tokens=200, repetition_penalty=1.2)
        | StrOutputParser()
    )

    # 添加后处理函数，去除可能的前缀
    def post_process(answer: str) -> str:
        prefixes = ["相关知识：", "答案：", "答：", "相关：", "知识："]
        for p in prefixes:
            if answer.startswith(p):
                answer = answer[len(p):].strip()
        return answer

    from langchain_core.runnables import RunnableLambda
    rag_chain = base_chain | RunnableLambda(post_process)
    app.state.rag_chain = rag_chain
    log_info("LangChain 链初始化完成")

    log_info("应用启动完成，各模块已初始化")
    yield
    app.state.llm_engine.unload()
    app.state.retrieval_engine.close()
    log_info("应用关闭，资源已释放")

def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(
        title="招投标智能问答系统",
        description="基于 RAG 的招投标领域问答服务",
        version="1.0.0",
        debug=config.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = config
    app.include_router(routes.router, prefix="/api/v1")
    return app

if __name__ == "__main__":
    import uvicorn
    config = AppConfig.from_env()
    print("配置加载成功")
    app = create_app(config)
    print(f"应用实例创建成功: {app}")
    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"uvicorn.run 异常: {e}")
        import traceback
        traceback.print_exc()
    print("uvicorn.run 已退出")