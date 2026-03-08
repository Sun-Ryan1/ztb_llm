from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from core_modules.retrieval import RetrievalEngine
from core_modules.utils import select_best_document
from pydantic import Field


class CustomRetriever(BaseRetriever):
    """将现有的 RetrievalEngine 适配为 LangChain 的 Retriever，只返回最佳文档"""
    
    retrieval_engine: RetrievalEngine = Field(..., description="The retrieval engine instance")

    def __init__(self, retrieval_engine: RetrievalEngine):
        super().__init__(retrieval_engine=retrieval_engine)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        # 先获取所有检索结果
        results = self.retrieval_engine.query(
            query_text=query,
            top_k=kwargs.get("top_k", 5),
            return_format="list"
        )
        # 使用 select_best_document 选择最佳文档
        best = select_best_document(results, query)
        if best:
            doc = Document(
                page_content=best["content"],
                metadata={
                    "id": best.get("id"),
                    "similarity": best.get("similarity"),
                    **best.get("metadata", {})
                }
            )
            return [doc]
        return []

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)