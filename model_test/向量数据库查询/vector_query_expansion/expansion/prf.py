# -*- coding: utf-8 -*-
"""
伪相关反馈 (Pseudo Relevance Feedback) 扩展器
依赖：jieba, collections.Counter
注意：需要传入检索器实例，以进行首次检索
"""

from collections import Counter
from typing import List, Optional

import jieba
import jieba.analyse

from .base import BaseQueryExpander


class PseudoRelevanceFeedbackExpander(BaseQueryExpander):
    """
    伪相关反馈扩展器
    对原始查询进行首次检索，从 top_k 篇文档中提取高频关键词，添加到查询中。
    """

    def __init__(
        self,
        retriever,  # OptimizedVectorDBQuery 实例
        top_k: int = 3,
        top_terms: int = 5,
        min_term_freq: int = 1,
        use_tfidf: bool = True,
    ):
        """
        Args:
            : 检索器实例，必须具有 _hybrid_search 方法
            top_k: 
            top_terms: 提取的关键词数量
            min_term_freq: 
            use_tfidf: 是否使用 TF-IDF 提取关键词，否则使用词频统计
        """
        self.retriever = retriever
        self.top_k = top_k
        self.top_terms = top_terms
        self.min_term_freq = min_term_freq
        self.use_tfidf = use_tfidf

    def expand(self, query: str, **kwargs) -> str:
        # 1. 首次检索
        results = self.retriever._hybrid_search(
            query, self.top_k, min_similarity=0.0
        )
        if not results:
            return query

        # 2. 提取关键词
        expansion_terms = self._extract_terms(results)

        # 3. 拼接扩展词
        if expansion_terms:
            return query + " " + " ".join(expansion_terms)
        return query

    def _extract_terms(self, docs: List[dict]) -> List[str]:
        """从文档列表中提取高频词"""
        if self.use_tfidf:
            # 合并所有文档内容
            all_text = " ".join([doc["content"] for doc in docs])
            keywords = jieba.analyse.extract_tags(
                all_text, topK=self.top_terms, withWeight=False
            )
            return keywords
        else:
            # 简单词频统计
            counter = Counter()
            for doc in docs:
                tokens = jieba.lcut(doc["content"])
                tokens = [t for t in tokens if len(t) >= 2]
                counter.update(tokens)
            # 过滤低于最小词频的词
            filtered = [
                word for word, count in counter.items()
                if count >= self.min_term_freq
            ]
            return filtered[: self.top_terms]

    def get_expansion_terms(self, query: str, **kwargs) -> List[str]:
        results = self.retriever._hybrid_search(
            query, self.top_k, min_similarity=0.0
        )
        if not results:
            return []
        return self._extract_terms(results)

    def reset(self) -> None:
        """PRF 无持久状态，无需操作"""
        pass