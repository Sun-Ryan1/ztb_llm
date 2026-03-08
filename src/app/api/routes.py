from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import logging
import re

from core_modules.dependencies import get_llm_engine, get_retrieval_engine, get_prompt_manager
from core_modules.retrieval import RetrievalEngine
from core_modules.llm_integration import LLMEngine
from core_modules.prompt_engineering import PromptManager, format_context
from core_modules.utils import select_best_document, extract_company_name

router = APIRouter()
logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    use_rerank: Optional[bool] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    query_type: str
    retrieval_time: float
    generation_time: float
    total_time: float

def clean_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'(?<=[\u4e00-\u9fa5]) (?=[\u4e00-\u9fa5])', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
    llm_engine: LLMEngine = Depends(get_llm_engine),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
):
    start_total = time.time()

    # 问候语处理
    greeting_phrases = ["你是谁", "你叫什么", "介绍一下你自己", "介绍下你自己", "自我介绍", "你好", "您好", "hello", "hi"]
    query_clean = request.query.strip().lower()
    is_greeting = False
    if query_clean in [p.lower() for p in greeting_phrases]:
        is_greeting = True
    elif len(query_clean) < 10 and any(phrase.lower() in query_clean for phrase in greeting_phrases):
        is_greeting = True

    if is_greeting:
        prompt = "请用中文简单地介绍一下你自己，说明你是招投标智能助手，可以帮助用户查询公司信息、产品价格、中标项目、法规条款等。回答要友好自然。"
        gen_start = time.time()
        temperature = request.temperature if request.temperature is not None else 0.5
        raw_answer = llm_engine.generate(
            prompt,
            max_new_tokens=150,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.0
        )
        generation_time = time.time() - gen_start
        answer = clean_answer(raw_answer)
        return AskResponse(
            answer=answer,
            retrieved_documents=[],
            query_type="greeting",
            retrieval_time=0,
            generation_time=generation_time,
            total_time=time.time() - start_total
        )

    retrieval_start = time.time()
    docs = retrieval_engine.query(
        query_text=request.query,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
        return_format="list"
    )
    retrieval_time = time.time() - retrieval_start

    best_doc = select_best_document(docs, request.query)
    use_direct_answer = best_doc is not None
    answer = None
    generation_time = 0.0

    if use_direct_answer:
        answer = clean_answer(best_doc['content'])
        logger.info(f"直接使用文档作为答案: {answer[:100]}...")
    else:
        is_spec_query = any(kw in request.query for kw in ["规格", "型号", "参数"])
        if is_spec_query:
            answer = "知识库中暂无该产品的规格信息，建议咨询供应商或查阅产品说明书。"
        else:
            answer = "知识库中暂无相关信息，请尝试更精确的查询。"
        logger.info("无直接匹配文档，返回缺省提示")

    total_time = time.time() - start_total
    docs_sorted = sorted(docs, key=lambda x: x.get('similarity', 0), reverse=True)

    return AskResponse(
        answer=answer,
        retrieved_documents=docs_sorted,
        query_type="general",
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time
    )

@router.get("/health")
async def health():
    return {"status": "ok"}

class LangChainAskRequest(AskRequest):
    pass

@router.post("/ask_langchain", response_model=AskResponse)
async def ask_langchain(
    http_request: Request,
    req_body: LangChainAskRequest,
    retrieval_engine: RetrievalEngine = Depends(get_retrieval_engine),
):
    """使用 LangChain 链的问答接口（实验性）
    """
    rag_chain = http_request.app.state.rag_chain

    start_total = time.time()

    retrieval_start = time.time()
    docs = retrieval_engine.query(
        query_text=req_body.query,
        top_k=req_body.top_k,
        use_rerank=req_body.use_rerank,
        return_format="list"
    )
    retrieval_time = time.time()
"""
    使用 LangChain 的检索器直接返回最佳文档，不经过 LLM 生成（确定性）
    """
    start_total = time.time()

    retrieval_start = time.time()
    docs = retrieval_engine.query(
        query_text=req_body.query,
        top_k=req_body.top_k,
        use_rerank=req_body.use_rerank,
        return_format="list"
    )
    retrieval_time = time.time() - retrieval_start

    best_doc = select_best_document(docs, req_body.query)
    if best_doc:
        answer = clean_answer(best_doc['content'])
        generation_time = 0.0
    else:
        answer = "知识库中暂无相关信息，请尝试更精确的查询。"
        generation_time = 0.0

    total_time = time.time() - start_total
    docs_sorted = sorted(docs, key=lambda x: x.get('similarity', 0), reverse=True)

    return AskResponse(
        answer=answer,
        retrieved_documents=docs_sorted,
        query_type="general",
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time
    )