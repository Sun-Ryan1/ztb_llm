from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """对候选文档重排序，返回 top_k 结果"""
        pass