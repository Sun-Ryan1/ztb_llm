# -*- coding: utf-8 -*-
"""
查询扩展器抽象基类
所有具体的扩展策略必须继承此类并实现 expand 和 get_expansion_terms 方法。
"""

from abc import ABC, abstractmethod
from typing import List


class BaseQueryExpander(ABC):
    """查询扩展器基类"""

    @abstractmethod
    def expand(self, query: str, **kwargs) -> str:
        """
        输入原始查询，返回扩展后的查询字符串。
        扩展后的字符串将直接用于检索。
        """
        pass

    @abstractmethod
    def get_expansion_terms(self, query: str, **kwargs) -> List[str]:
        """
        返回本次扩展产生的扩展词列表（用于分析、调试、日志）。
        """
        pass

    def reset(self) -> None:
        """
        可选：重置扩展器状态（例如伪相关反馈可能需要重置缓存）
        """
        pass