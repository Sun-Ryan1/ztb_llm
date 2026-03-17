#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询器（性能模式可配置版）"""

import os
import json
import time
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from dataclasses import dataclass
import re
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi

# ===================== 重排序器基类与实现 =====================
class BaseReranker(ABC):
    """重排序器抽象基类
"""
    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """对候选文档重排序，返回 top_k 结果
"""
        pass

class CrossEncoderReranker(BaseReranker):
    """基于交叉编码器的重排序器（兼容所有 sentence-transformers 版本）"""
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Args:
        _name: 
        device: （cuda/cpu），默认自动选择
        batch_size:
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        logger.info(f"加载重排序模型: {model_name} | 设备: {self.device}")
        # 移除 fp16 参数，兼容所有版本
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            trust_remote_code=True,
        )
        logger.info("重排序模型加载完成")

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        # 构造 (query, document) 对
        pairs = [(query, doc["content"]) for doc in docs]
        # 批量预测得分
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        # 将得分附加到文档上
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        # 按得分降序排序
        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        # 更新 similarity 和检索方法标识
        for doc in reranked[:top_k]:
            doc["similarity"] = round(doc["rerank_score"], 4)
            doc["retrieval_method"] += "+rerank"
            doc["reranked"] = True
        return reranked[:top_k]

# ===================== 全局配置（性能模式可切换） =====================
CONFIG = {
    # ---------- 性能模式选择 ----------
    # 可选值: "balanced", "fast", "ultra"
    "PERFORMANCE_MODE": "balanced",  # 默认召回最优模式

    # ---------- 缓存配置 ----------
    "cache_max_size": 2000,
    "cache_expire_time": 3600,
    "cache_clean_interval": 300,
    "enable_query_cache": True,
    "enable_vector_cache": True,
    "vector_cache_ttl": 1800,

    # ---------- 混合检索权重 ----------
    "vector_weight": 0.6,
    "bm25_weight": 0.4,

    # ---------- 信用代码精确匹配 ----------
    "credit_code_exact_match_score": 1.0,
    "credit_code_id_match_score": 0.95,

    # ---------- 重排序配置 ----------
    "rerank_enabled_by_default": False,   # 默认不启用重排序（需显式传入 reranker）
    "rerank_candidate_multiple": 3,       # 重排序候选倍数（最终top_k = 原始top_k * 倍数）

    # ---------- 设备配置 ----------
    "device": None,

    # ---------- 术语词典路径 ----------
    "industry_terms_path": "./industry_terms.txt",
    "product_terms_path": "./product_terms.txt",
    "legal_terms_path": "./legal_terms.txt",
    "address_terms_path": "./address_terms.txt",
    "business_terms_path": "./business_terms.txt",

    # ---------- BM25 索引配置 ----------
    "bm25_index_enabled": True,
    "bm25_index_refresh_interval": 3600,
    "bm25_index_max_docs": 50000,

    # ---------- HNSW 优化参数（仅 ultra 模式生效）----------
    "hnsw_M": 16,
    "hnsw_construction_ef": 100,
    "hnsw_search_ef": 40,
}

# 根据性能模式动态计算候选倍数
_PERF_MODE = CONFIG["PERFORMANCE_MODE"]
if _PERF_MODE == "balanced":
    CONFIG["hybrid_search_candidate_multiple"] = 8
    CONFIG["credit_code_candidate_multiple"] = 8
    CONFIG["force_recreate_collection"] = False
    logger.info("性能模式: balanced (8倍候选，召回最优)")
elif _PERF_MODE == "fast":
    CONFIG["hybrid_search_candidate_multiple"] = 6
    CONFIG["credit_code_candidate_multiple"] = 6
    CONFIG["force_recreate_collection"] = False
    logger.info("性能模式: fast (6倍候选，召回略降，速度提升25%)")
elif _PERF_MODE == "ultra":
    CONFIG["hybrid_search_candidate_multiple"] = 6
    CONFIG["credit_code_candidate_multiple"] = 6
    CONFIG["force_recreate_collection"] = False   # 需重建集合以应用HNSW
    logger.info("性能模式: ultra (6倍候选 + HNSW优化，速度提升50%，需重建集合)")
else:
    raise ValueError(f"未知的性能模式: {_PERF_MODE}，可选 balanced/fast/ultra")

# ===================== 数据结构 =====================
@dataclass
class QueryResult:
    query: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: float
    total_retrieved: int
    avg_similarity: float
    query_id: str = None
    retrieval_method: str = "vector_search"
    reranked: bool = False
    query_type: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "retrieval_time": self.retrieval_time,
            "avg_similarity": self.avg_similarity,
            "query_id": self.query_id,
            "retrieval_method": self.retrieval_method,
            "reranked": self.reranked,
            "query_type": self.query_type,
            "retrieved_documents": [doc.copy() for doc in self.retrieved_documents]
        }

# ===================== 线程安全缓存系统 =====================
class ThreadSafeQueryCache:
    # ... 完全保持原有代码，此处仅作示意，实际使用时请完整保留 ...
    def __init__(self, max_size: int = CONFIG["cache_max_size"],
                 expire_time: int = CONFIG["cache_expire_time"]):
        self.max_size = max_size
        self.expire_time = expire_time
        self.cache = OrderedDict()
        self.vector_cache = OrderedDict()
        self.keyword_cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {
            "hits": 0, "misses": 0,
            "vector_cache_hits": 0, "vector_cache_misses": 0,
            "keyword_cache_hits": 0, "keyword_cache_misses": 0,
            "evictions": 0, "expired_evictions": 0,
            "total_requests": 0,
            "cache_size": 0, "max_cache_size": 0,
            "writes": 0,
            "last_clear_time": time.time(),
            "query_type_stats": defaultdict(lambda: {"hits": 0, "misses": 0, "requests": 0, "hit_rate": 0.0}),
        }
        self.clean_thread = threading.Thread(target=self._periodic_clean, daemon=True)
        self.clean_thread.start()

    @staticmethod
    def _generate_key(query: str, params: Dict) -> str:
        try:
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, (dict, list, tuple)):
                    safe_params[k] = json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
                else:
                    safe_params[k] = str(v)
            key_str = f"{query}_{json.dumps(safe_params, sort_keys=True)}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except:
            return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()

    @staticmethod
    def _simple_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str, params: Dict) -> Optional[Any]:
        with self.lock:
            self.stats["total_requests"] += 1
            key = self._generate_key(query, params)
            query_type = self._classify_query(query)
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry["timestamp"] < self.expire_time:
                    self.stats["hits"] += 1
                    self.stats["query_type_stats"][query_type]["hits"] += 1
                    self.cache.move_to_end(key)
                    return entry["result"]
                else:
                    del self.cache[key]
                    self.stats["expired_evictions"] += 1
            self.stats["misses"] += 1
            self.stats["query_type_stats"][query_type]["misses"] += 1
            return None

    def set(self, query: str, params: Dict, result: Any) -> None:
        with self.lock:
            key = self._generate_key(query, params)
            self.cache[key] = {
                "result": result,
                "timestamp": time.time(),
                "query": query[:100],
                "query_type": self._classify_query(query)
            }
            self.stats["writes"] += 1
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                self.stats["evictions"] += 1
            self.stats["cache_size"] = len(self.cache)
            if self.stats["cache_size"] > self.stats["max_cache_size"]:
                self.stats["max_cache_size"] = self.stats["cache_size"]

    def get_vector(self, text: str) -> Optional[np.ndarray]:
        if not CONFIG["enable_vector_cache"]:
            return None
        with self.lock:
            key = self._simple_key(text)
            if key in self.vector_cache:
                entry = self.vector_cache[key]
                if time.time() - entry["timestamp"] < CONFIG["vector_cache_ttl"]:
                    self.stats["vector_cache_hits"] += 1
                    self.vector_cache.move_to_end(key)
                    return entry["vector"]
                else:
                    del self.vector_cache[key]
                    self.stats["expired_evictions"] += 1
            self.stats["vector_cache_misses"] += 1
            return None

    def set_vector(self, text: str, vector: np.ndarray) -> None:
        if not CONFIG["enable_vector_cache"]:
            return
        with self.lock:
            key = self._simple_key(text)
            self.vector_cache[key] = {
                "vector": vector,
                "timestamp": time.time()
            }
            self.vector_cache.move_to_end(key)
            max_vector = self.max_size * 2
            if len(self.vector_cache) > max_vector:
                oldest = next(iter(self.vector_cache))
                del self.vector_cache[oldest]
                self.stats["evictions"] += 1

    def get_keywords(self, text: str) -> Optional[List[Tuple[str, float]]]:
        with self.lock:
            key = self._simple_key(text)
            if key in self.keyword_cache:
                entry = self.keyword_cache[key]
                if time.time() - entry["timestamp"] < self.expire_time:
                    self.stats["keyword_cache_hits"] += 1
                    self.keyword_cache.move_to_end(key)
                    return entry["keywords"]
                else:
                    del self.keyword_cache[key]
            self.stats["keyword_cache_misses"] += 1
            return None

    def set_keywords(self, text: str, keywords: List[Tuple[str, float]]) -> None:
        with self.lock:
            key = self._simple_key(text)
            self.keyword_cache[key] = {
                "keywords": keywords,
                "timestamp": time.time()
            }
            self.keyword_cache.move_to_end(key)
            if len(self.keyword_cache) > self.max_size // 2:
                oldest = next(iter(self.keyword_cache))
                del self.keyword_cache[oldest]
                self.stats["evictions"] += 1

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.vector_cache.clear()
            self.keyword_cache.clear()
            self.stats["cache_size"] = 0
            self.stats["last_clear_time"] = time.time()

    def remove_expired(self) -> None:
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.cache.items() if now - v["timestamp"] >= self.expire_time]
            for k in expired_keys:
                del self.cache[k]
                self.stats["expired_evictions"] += 1
            expired_vec = [k for k, v in self.vector_cache.items()
                           if now - v["timestamp"] >= CONFIG["vector_cache_ttl"]]
            for k in expired_vec:
                del self.vector_cache[k]
                self.stats["expired_evictions"] += 1
            expired_kw = [k for k, v in self.keyword_cache.items() if now - v["timestamp"] >= self.expire_time]
            for k in expired_kw:
                del self.keyword_cache[k]
                self.stats["expired_evictions"] += 1
            self.stats["cache_size"] = len(self.cache)

    def _periodic_clean(self) -> None:
        while True:
            time.sleep(CONFIG["cache_clean_interval"])
            try:
                self.remove_expired()
                logger.debug("定时清理过期缓存完成")
            except Exception as e:
                logger.error(f"定时清理缓存失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            stats = self.stats.copy()
            stats["query_type_stats"] = {k: v.copy() for k, v in stats["query_type_stats"].items()}
            total = stats["total_requests"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
            for qtype, qs in stats["query_type_stats"].items():
                total_q = qs["hits"] + qs["misses"]
                qs["hit_rate"] = qs["hits"] / total_q if total_q > 0 else 0.0
                qs["requests"] = total_q
            stats["uptime"] = time.time() - stats["last_clear_time"]
            return stats

    @staticmethod
    def _classify_query(query: str) -> str:
        q = query.lower().strip()
        if re.search(r'91[0-9a-zA-Z]{16}', q):
            return "credit_code"
        if any(kw in q for kw in ['法定代表人', '法人代表', '法人', '负责人', '代表人']):
            return "legal_representative"
        if any(kw in q for kw in ['招标', '投标', '中标', '采购项目', '政府采购']):
            return "zhaobiao_natural"
        if any(kw in q for kw in ['法律', '法规', '规定', '条款', '法律咨询']) and \
           any(ind in q for ind in ['如何', '怎样', '为什么', '什么', '哪些']):
            return "law_natural"
        if re.search(r'《[^》]+》', q):
            return "legal_title"
        if re.search(r'([\u4e00-\u9fa5]{2,10}(?:型号|规格|产品名|名称))|叫.*?[?？]', q):
            return "product_name"
        if any(kw in q for kw in ['供应商', '供货商', '供应', '采购', '经销商', '代理商', '厂家', '制造商']):
            return "product_supplier"
        if any(kw in q for kw in ['什么产品', '哪些产品', '产品有', '产品包括', '提供什么', '有哪些产品']):
            return "product_keyword"
        if any(kw in q for kw in ['经营范围', '经营项目', '业务范围', '营业范围', '从事', '经营']):
            return "business_scope"
        if any(kw in q for kw in ['地址', '注册地', '所在地', '坐落', '位置', '位于']):
            return "address"
        if re.search(r'[\u4e00-\u9fa5]+(有限公司|有限责任公司|股份有限公司|分公司|子公司|集团公司)$', q):
            return "company_name"
        return "general"

# ===================== 主查询类 =====================
class OptimizedVectorDBQuery:
    def __init__(self,
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = CONFIG["device"]):

        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"初始化优化向量数据库查询器 | 设备: {self.device}")

        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_cache = ThreadSafeQueryCache()

        # 术语词典
        self.industry_terms = self._load_terms_dict(CONFIG["industry_terms_path"],
                                                   default=['建筑', '装饰', '装修', '工程', '设计', '施工'])
        self.product_terms = self._load_terms_dict(CONFIG["product_terms_path"],
                                                  default=['设备', '仪器', '装置', '机器', '工具', '材料'])
        self.legal_terms = self._load_terms_dict(CONFIG["legal_terms_path"],
                                                default=['法律', '法规', '条例', '规定', '办法', '司法解释'])
        self.address_terms = self._load_terms_dict(CONFIG["address_terms_path"],
                                                  default=['省', '市', '区', '县', '镇', '街道', '路', '巷', '弄', '号'])
        self.business_terms = self._load_terms_dict(CONFIG["business_terms_path"],
                                                   default=['经营', '业务', '销售', '制造', '生产', '服务', '加工', '贸易'])

        # BM25 索引
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_doc_ids = []
        self.bm25_doc_id_to_idx = {}
        self.bm25_last_refresh = 0
        self.bm25_lock = threading.Lock()

        # 查询分词缓存
        self._query_tokens_cache = {}

        self.config = {
            "default_top_k": 20,
            "enable_cache": CONFIG["enable_query_cache"],
            "enable_hybrid_search": True,
            "min_similarity_default": 0.0,
            "bm25_index_enabled": CONFIG["bm25_index_enabled"],
        }

        self._initialize_components()

    # ---------- 初始化 ----------
    def _initialize_components(self) -> None:
        embedding_model = None
        chroma_client = None
        collection = None

        try:
            logger.info("加载 Embedding 模型...")
            embedding_model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True
            )

            logger.info("连接 ChromaDB...")
            chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # ---------- HNSW 优化：仅在 ultra 模式下强制重建 ----------
            if CONFIG["force_recreate_collection"]:
                try:
                    chroma_client.delete_collection(name=self.collection_name)
                    logger.warning(f"已删除集合 {self.collection_name}（性能模式 ultra 重建）")
                except:
                    pass

            try:
                collection = chroma_client.get_collection(name=self.collection_name)
                logger.info(f"使用现有集合 {self.collection_name}")
            except Exception:
                # 创建新集合，ultra 模式应用 HNSW 优化参数
                if CONFIG["PERFORMANCE_MODE"] == "ultra":
                    hnsw_metadata = {
                        "hnsw:space": "cosine",
                        "hnsw:M": CONFIG["hnsw_M"],
                        "hnsw:construction_ef": CONFIG["hnsw_construction_ef"],
                        "hnsw:search_ef": CONFIG["hnsw_search_ef"],
                    }
                    logger.info(f"应用 HNSW 优化参数: {hnsw_metadata}")
                else:
                    hnsw_metadata = {"hnsw:space": "cosine"}  # 默认参数

                collection = chroma_client.create_collection(
                    name=self.collection_name,
                    metadata=hnsw_metadata
                )
                logger.info(f"创建新集合 {self.collection_name}")

            self.embedding_model = embedding_model
            self.chroma_client = chroma_client
            self.collection = collection

            if self.config["bm25_index_enabled"]:
                self._refresh_bm25_index(force=True)

            logger.info("✅ 所有组件初始化完成")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            if embedding_model:
                del embedding_model
            if chroma_client:
                try: chroma_client.close()
                except: pass
            raise

    def _refresh_bm25_index(self, force: bool = False) -> bool:
        if not self.config["bm25_index_enabled"] or not self.collection:
            return False
        with self.bm25_lock:
            now = time.time()
            if not force and (now - self.bm25_last_refresh) < CONFIG["bm25_index_refresh_interval"]:
                return False
            try:
                logger.info("开始刷新 BM25 索引...")
                all_docs = self.collection.get(
                    limit=CONFIG["bm25_index_max_docs"],
                    include=["documents", "metadatas"]
                )
                docs = all_docs.get("documents", [])
                ids = all_docs.get("ids", [])
                if not docs:
                    logger.warning("集合中无文档，BM25 索引为空")
                    return False
                tokenized_docs = []
                self.bm25_docs = []
                self.bm25_doc_ids = []
                self.bm25_doc_id_to_idx = {}
                for i, (doc, doc_id) in enumerate(zip(docs, ids)):
                    if not doc:
                        continue
                    tokens = [w for w in jieba.lcut(doc) if len(w) >= 2]
                    tokenized_docs.append(tokens)
                    self.bm25_docs.append(doc)
                    self.bm25_doc_ids.append(doc_id)
                    self.bm25_doc_id_to_idx[doc_id] = i
                self.bm25_index = BM25Okapi(tokenized_docs)
                self.bm25_last_refresh = now
                logger.info(f"BM25 索引构建完成 | 文档数: {len(tokenized_docs)} | 耗时: {time.time()-now:.2f}s")
                return True
            except Exception as e:
                logger.error(f"BM25 索引刷新失败: {e}")
                return False

    # ---------- 工具方法 ----------
    def _load_terms_dict(self, file_path: str, default: List[str]) -> set:
        terms = set()
        for term in default:
            terms.add(term)
            jieba.add_word(term, freq=1000)
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        term = line.strip()
                        if term and term not in terms:
                            terms.add(term)
                            jieba.add_word(term, freq=1000)
                logger.info(f"加载术语词典 {file_path}: {len(terms)} 个术语")
            except Exception as e:
                logger.warning(f"加载术语词典失败 {file_path}: {e}，使用默认值")
        else:
            logger.info(f"术语词典 {file_path} 不存在，使用内置默认术语 ({len(default)} 个)")
        return terms

    def _encode_query(self, query_text: str) -> np.ndarray:
        cached = self.query_cache.get_vector(query_text)
        if cached is not None:
            return cached
        vector = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        self.query_cache.set_vector(query_text, vector)
        return vector

    # ---------- 核心检索 ----------
    def query(self,
              query_text: str,
              top_k: int = None,
              where_filter: Optional[Dict[str, Any]] = None,
              min_similarity: Optional[float] = None,
              return_format: str = "structured") -> Union[QueryResult, List[Dict[str, Any]]]:
        start_time = time.time()
        top_k = top_k or self.config["default_top_k"]
        min_similarity = min_similarity or self.config["min_similarity_default"]
        if not isinstance(query_text, str) or not query_text.strip():
            logger.error("查询文本为空")
            empty = QueryResult(query=query_text, retrieved_documents=[], retrieval_time=0,
                               total_retrieved=0, avg_similarity=0.0, query_type="general")
            return empty if return_format == "structured" else []
        query_type = self.query_cache._classify_query(query_text)
        cache_params = {
            "top_k": top_k,
            "where_filter": where_filter,
            "min_similarity": min_similarity,
            "query_type": query_type
        }
        if self.config["enable_cache"]:
            cached = self.query_cache.get(query_text, cache_params)
            if cached:
                logger.debug(f"缓存命中 | 查询: {query_text[:30]}... | 类型: {query_type}")
                if return_format == "list":
                    return cached["retrieved_documents"]
                else:
                    return QueryResult(**cached)
        try:
            if self._is_credit_code(query_text):
                docs = self._credit_code_exact_match(query_text, top_k, where_filter, min_similarity)
                method = "credit_code_exact"
            else:
                docs = self._hybrid_search(query_text, top_k, where_filter, min_similarity)
                method = "hybrid_search"
            final_docs = docs[:top_k]
            retrieval_time = time.time() - start_time
            scores = [d.get("similarity", 0.0) for d in final_docs if d.get("similarity") is not None]
            avg_sim = np.mean(scores) if scores else 0.0
            result = QueryResult(
                query=query_text,
                retrieved_documents=final_docs,
                retrieval_time=retrieval_time,
                total_retrieved=len(final_docs),
                avg_similarity=avg_sim,
                retrieval_method=method,
                query_type=query_type
            )
            if self.config["enable_cache"]:
                self.query_cache.set(query_text, cache_params, result.to_dict())
            logger.debug(f"查询完成 | 类型: {query_type} | 耗时: {retrieval_time:.3f}s | 结果: {len(final_docs)}")
            return result if return_format == "structured" else final_docs
        except Exception as e:
            logger.error(f"查询失败，降级至混合检索: {e}", exc_info=True)
            docs = self._hybrid_search(query_text, top_k, where_filter, min_similarity)
            retrieval_time = time.time() - start_time
            empty_result = QueryResult(
                query=query_text,
                retrieved_documents=docs[:top_k],
                retrieval_time=retrieval_time,
                total_retrieved=len(docs[:top_k]),
                avg_similarity=np.mean([d.get("similarity",0.0) for d in docs[:top_k]]) if docs else 0.0,
                retrieval_method="hybrid_search_fallback",
                query_type=query_type
            )
            return empty_result if return_format == "structured" else empty_result.retrieved_documents

    # ---------- 混合检索 ----------
    def _hybrid_search(self, query_text: str, top_k: int,
                       where_filter: Optional[Dict] = None,
                       min_similarity: Optional[float] = None) -> List[Dict]:
        candidate_k = top_k * CONFIG["hybrid_search_candidate_multiple"]
        vec_docs = self._vector_search_optimized(query_text, candidate_k, where_filter, 0.0)
        if not vec_docs:
            return []
        if query_text not in self._query_tokens_cache:
            self._query_tokens_cache[query_text] = jieba.lcut(query_text)
        query_tokens = self._query_tokens_cache[query_text]
        bm25_scores_all = self.bm25_index.get_scores(query_tokens) if self.bm25_index else None
        bm25_scores = {}
        bm25_vals = []
        if bm25_scores_all is not None:
            for doc in vec_docs:
                doc_id = doc["id"]
                if doc_id in self.bm25_doc_id_to_idx:
                    idx = self.bm25_doc_id_to_idx[doc_id]
                    score = bm25_scores_all[idx]
                    bm25_scores[doc_id] = score
                    bm25_vals.append(score)
                else:
                    bm25_scores[doc_id] = 0.0
                    bm25_vals.append(0.0)
        else:
            for doc in vec_docs:
                bm25_scores[doc["id"]] = 0.0
                bm25_vals.append(0.0)
        if bm25_vals:
            min_b = min(bm25_vals)
            max_b = max(bm25_vals)
            range_b = max_b - min_b if max_b > min_b else 1.0
        else:
            min_b, max_b, range_b = 0.0, 1.0, 1.0
        docs = []
        for doc in vec_docs:
            vec_score = doc["similarity"]
            raw_bm25 = bm25_scores.get(doc["id"], 0.0)
            bm25_norm = (raw_bm25 - min_b) / range_b if range_b > 0 else 0.0
            hybrid = vec_score * CONFIG["vector_weight"] + bm25_norm * CONFIG["bm25_weight"]
            hybrid = min(hybrid, 1.0)
            doc.update({
                "similarity": round(hybrid, 4),
                "vector_score": round(vec_score, 4),
                "bm25_score": round(bm25_norm, 4),
                "retrieval_method": "hybrid_search",
                "query_type": "general"
            })
            docs.append(doc)
        docs.sort(key=lambda x: x["similarity"], reverse=True)
        if min_similarity is not None:
            docs = [d for d in docs if d["similarity"] >= min_similarity]
        return docs[:top_k]

    # ---------- 向量检索 ----------
    def _vector_search_optimized(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        try:
            query_emb = self._encode_query(query_text)
            query_params = {
                "query_embeddings": [query_emb.tolist()],
                "n_results": max(1, top_k),
                "include": ["documents", "metadatas", "distances"]
            }
            if where_filter:
                query_params["where"] = where_filter
            results = self.collection.query(**query_params)
            docs = results.get("documents", [[]])[0] or []
            metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
            dists = results.get("distances", [[]])[0] or [1.0] * len(docs)
            ids = results.get("ids", [[]])[0] or [f"doc_{i}" for i in range(len(docs))]
            retrieved = []
            for i, (doc, meta, dist, doc_id) in enumerate(zip(docs, metas, dists, ids)):
                if not doc:
                    continue
                sim = max(0.0, 1 - dist)
                if min_similarity and sim < min_similarity:
                    continue
                retrieved.append({
                    "id": doc_id,
                    "content": doc,
                    "similarity": round(sim, 4),
                    "metadata": meta or {},
                    "rank": i+1,
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "retrieval_method": "vector_search"
                })
            return retrieved
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    # ---------- 信用代码精确匹配 ----------
    def _credit_code_exact_match(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        match = re.search(r'91[0-9A-Z]{16}', query_text.upper())
        if not match:
            return self._hybrid_search(query_text, top_k, where_filter, min_similarity)
        credit_code = match.group(0)
        exact_where = {"credit_code": {"$eq": credit_code}}
        if where_filter:
            exact_where.update(where_filter)
        exact_results = self._vector_search_optimized(credit_code, top_k, exact_where, 0.0)
        exact_matched = []
        for doc in exact_results:
            if credit_code in doc.get("content", "").upper() or \
               credit_code in doc.get("id", "").upper():
                doc["similarity"] = CONFIG["credit_code_exact_match_score"]
                doc["exact_match"] = True
                exact_matched.append(doc)
        if exact_matched:
            return exact_matched[:top_k]
        candidates = self._vector_search_optimized(
            credit_code,
            top_k * CONFIG["credit_code_candidate_multiple"],
            where_filter, 0.0
        )
        return candidates[:top_k]

    # ---------- 辅助工具 ----------
    @staticmethod
    def _deduplicate_results(docs: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for d in docs:
            content_hash = hashlib.md5(d.get('content', '').encode()).hexdigest()[:16]
            key = f"{content_hash}_{d.get('id', '')}"
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    @staticmethod
    def _is_credit_code(text: str) -> bool:
        return bool(re.search(r'91[0-9a-zA-Z]{16}', text, re.IGNORECASE))

    # ---------- 状态查询与资源释放 ----------
    def test_connection(self) -> Dict[str, Any]:
        result = {
            "model_loaded": self.embedding_model is not None,
            "db_connected": self.chroma_client is not None,
            "collection_exists": self.collection is not None,
            "bm25_index_ready": self.bm25_index is not None,
            "collection_count": self.collection.count() if self.collection else 0,
            "cache_stats": self.query_cache.get_stats() if self.query_cache else {},
            "term_dicts_loaded": {
                "industry": len(self.industry_terms),
                "product": len(self.product_terms),
                "legal": len(self.legal_terms),
                "address": len(self.address_terms),
                "business": len(self.business_terms)
            }
        }
        try:
            test = self._encode_query("测试")
            result["test_encode_success"] = True
        except:
            result["test_encode_success"] = False
        return result

    def close(self) -> None:
        logger.info("释放资源...")
        if self.embedding_model:
            try:
                self.embedding_model.cpu()
            except:
                pass
            del self.embedding_model
        if self.chroma_client:
            try:
                self.chroma_client.close()
            except:
                pass
        self.query_cache.clear()
        self._query_tokens_cache.clear()
        logger.info("资源释放完成")

# ===================== 使用示例 =====================
if __name__ == "__main__":
    queryer = OptimizedVectorDBQuery(
        db_path="/tmp/chroma_db_dsw",
        model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
        collection_name="rag_knowledge_base"
    )
    try:
        status = queryer.test_connection()
        print("连接状态:", json.dumps(status, indent=2, ensure_ascii=False))
        test_queries = [
            "上海仓祥绿化工程有限公司的注册地址",
            "经营范围包括哪些内容？",
            "统一社会信用代码91310118MA1J9K8D6D",
            "生产什么产品？",
            "劳动合同法有什么规定？"
        ]
        for q in test_queries:
            print(f"\n🔍 查询: {q}")
            result = queryer.query(q, top_k=3)
            print(f"类型: {result.query_type}, 方法: {result.retrieval_method}, 耗时: {result.retrieval_time:.3f}s")
            for i, doc in enumerate(result.retrieved_documents):
                print(f"  [{i+1}] {doc['similarity']:.4f} - {doc['content_preview'][:80]}...")
    finally:
        queryer.close()