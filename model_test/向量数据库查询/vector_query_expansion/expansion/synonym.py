# -*- coding: utf-8 -*-
"""
同义词扩展器
依赖：jieba
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional

import jieba

from .base import BaseQueryExpander


class SynonymExpander(BaseQueryExpander):
    """
    基于词典的同义词扩展器
    将查询中的每个词替换为自身 + 最多 N 个同义词。
    """

    def __init__(
        self,
        synonym_path: str = "./data/synonyms.txt",
        max_synonyms: int = 2,
        expand_all_tokens: bool = True,
    ):
        """
        Args:
            _path: 
            max_synonyms: 
            expand_all_tokens: ；若为False，仅扩展词典中存在的词
        """
        self.max_synonyms = max_synonyms
        self.expand_all_tokens = expand_all_tokens
        self.synonym_dict = self._load_synonyms(synonym_path)

    def _load_synonyms(self, path: str) -> Dict[str, List[str]]:
        """加载同义词词典，格式：词,同义词1,同义词2,..."""
        d = defaultdict(list)
        if not os.path.exists(path):
            return d
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    synonyms = [p.strip() for p in parts[1:] if p.strip()]
                    if synonyms:
                        d[word].extend(synonyms)
        return d

    def expand(self, query: str, **kwargs) -> str:
        tokens = jieba.lcut(query)
        expanded_tokens = []

        for token in tokens:
            expanded_tokens.append(token)
            if token in self.synonym_dict:
                synonyms = self.synonym_dict[token][: self.max_synonyms]
                expanded_tokens.extend(synonyms)
            elif self.expand_all_tokens:
                # 可选的默认行为：不做任何事
                pass

        return " ".join(expanded_tokens)

    def get_expansion_terms(self, query: str, **kwargs) -> List[str]:
        tokens = jieba.lcut(query)
        terms = []
        for token in tokens:
            if token in self.synonym_dict:
                terms.extend(self.synonym_dict[token][: self.max_synonyms])
        return list(set(terms))

    def add_synonyms(self, word: str, synonyms: List[str]) -> None:
        """动态添加同义词（可用于在线学习）"""
        self.synonym_dict[word].extend(synonyms)