# -*- coding: utf-8 -*-
"""
组合扩展器
将多个扩展器按顺序串联，前一个扩展器的输出作为下一个的输入。
"""

from typing import List

from .base import BaseQueryExpander


class ComposedExpander(BaseQueryExpander):
    """
    组合多个扩展器，按顺序执行。
    """

    def __init__(self, expanders: List[BaseQueryExpander]):
        """
        Args:
            : 扩展器实例列表，按执行顺序排列
        """
        self.expanders = expanders

    def expand(self, query: str, **kwargs) -> str:
        expanded = query
        for exp in self.expanders:
            expanded = exp.expand(expanded, **kwargs)
        return expanded

    def get_expansion_terms(self, query: str, **kwargs) -> List[str]:
        all_terms = []
        current_query = query
        for exp in self.expanders:
            # 获取当前扩展器针对当前查询产生的扩展词
            terms = exp.get_expansion_terms(current_query, **kwargs)
            all_terms.extend(terms)
            # 更新查询以用于后续扩展器（因为后续扩展器可能依赖扩展后的查询）
            current_query = exp.expand(current_query, **kwargs)
        return list(set(all_terms))

    def reset(self) -> None:
        for exp in self.expanders:
            exp.reset()