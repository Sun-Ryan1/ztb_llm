# -*- coding: utf-8 -*-
"""
从配置文件（yaml/json）动态构建扩展器实例
"""

import os
import importlib
from typing import Dict, Any, Optional

import yaml

from .base import BaseQueryExpander
from .synonym import SynonymExpander
from .prf import PseudoRelevanceFeedbackExpander
from .compose import ComposedExpander


def load_expander_from_config(
    config_path: str,
    retriever: Optional[object] = None,
) -> Optional[BaseQueryExpander]:
    """
    从 YAML 配置文件加载扩展器。
    示例配置见 configs/expansion_config.yaml
    """
    if not os.path.exists(config_path):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config.get("enabled", False):
        return None

    expander_config = config.get("expander", {})
    expander_type = expander_config.get("type")

    if expander_type == "synonym":
        return SynonymExpander(
            synonym_path=expander_config.get("synonym_path", "./data/synonyms.txt"),
            max_synonyms=expander_config.get("max_synonyms", 2),
            expand_all_tokens=expander_config.get("expand_all_tokens", True),
        )
    elif expander_type == "prf":
        if retriever is None:
            raise ValueError("PRF扩展器需要传入retriever实例")
        return PseudoRelevanceFeedbackExpander(
            retriever=retriever,
            top_k=expander_config.get("top_k", 3),
            top_terms=expander_config.get("top_terms", 5),
            min_term_freq=expander_config.get("min_term_freq", 1),
            use_tfidf=expander_config.get("use_tfidf", True),
        )
    elif expander_type == "compose":
        sub_expanders = []
        for sub_cfg in expander_config.get("expanders", []):
            # 递归构建子扩展器
            sub_config = {"expander": sub_cfg, "enabled": True}
            # 临时写入字典，重新调用本函数
            # 简便起见，直接构建
            if sub_cfg["type"] == "synonym":
                sub_expanders.append(
                    SynonymExpander(
                        synonym_path=sub_cfg.get("synonym_path", "./data/synonyms.txt"),
                        max_synonyms=sub_cfg.get("max_synonyms", 2),
                    )
                )
            elif sub_cfg["type"] == "prf":
                if retriever is None:
                    raise ValueError("组合扩展中的PRF需要retriever")
                sub_expanders.append(
                    PseudoRelevanceFeedbackExpander(
                        retriever=retriever,
                        top_k=sub_cfg.get("top_k", 3),
                        top_terms=sub_cfg.get("top_terms", 5),
                    )
                )
        return ComposedExpander(sub_expanders)
    else:
        raise ValueError(f"未知扩展器类型: {expander_type}")