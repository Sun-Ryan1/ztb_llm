from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import numpy as np

from .base import BaseReranker

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None, batch_size: int = 32):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
    
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        
        # 构造 query-doc 对
        pairs = [(query, doc["content"]) for doc in docs]
        
        # 批量计算相关性得分
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # 将得分附加到文档上
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        
        # 按得分降序排序
        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        
        # 更新 similarity 字段（可选）
        for doc in reranked[:top_k]:
            doc["similarity"] = round(doc["rerank_score"], 4)
            doc["retrieval_method"] = "hybrid_search + rerank"
            doc["reranked"] = True
        
        return reranked[:top_k]