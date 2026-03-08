#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询器（优化版）
优化点：
1. 补全所有缺失方法，修复语法错误
2. 增强缓存系统（线程安全、定时清理、向量缓存）
3. 重构检索逻辑（信用代码精确匹配优化、混合检索权重合理化）
4. 完善异常处理和资源管理
5. 配置解耦（硬编码参数抽离为可配置项）
6. 强化边界条件校验（空值、参数范围、类型校验）
7. 性能优化（向量缓存、重复计算减少、并发安全）
8. 新增test_connection方法，支持连接测试
"""

import os
import json
import time
import logging
import hashlib
import pickle
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import jieba
import jieba.analyse
from rank_bm25 import BM25Okapi

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s
"""查询结果数据结构
"""query: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: float
    total_retrieved: int
    avg_similarity: float
    query_id: str = None
    retrieval_method: str = "vector_search"
    reranked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（深拷贝避免原数据篡改）
"""
        return {
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "retrieval_time": self.retrieval_time,
            "avg_similarity": self.avg_similarity,
            "query_id": self.query_id,
            "retrieval_method": self.retrieval_method,
            "reranked": self.reranked,
            "retrieved_documents": [doc.copy() for doc in self.retrieved_documents]
        }

class ThreadSafeQueryCache:
    """线程安全的查询缓存系统（替代原QueryCache）
"""
    
    def __init__(self, max_size: int = CONFIG["cache_max_size"], expire_time: int = CONFIG["cache_expire_time"]):
        self.max_size = max_size
        self.expire_time = expire_time
        self.cache = OrderedDict()
        self.vector_cache = {}  # 新增：缓存查询向量，减少重复编码
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "cache_size": 0,
            "max_cache_size": 0,
            "writes": 0,
            "evictions": 0,
            "expired_evictions": 0,
            "hit_rate": 0.0,
            "last_clear_time": time.time(),
            "vector_cache_hits": 0,
            "vector_cache_misses": 0,
            "query_type_stats": defaultdict(lambda: {
                "hits": 0,
                "misses": 0,
                "requests": 0,
                "hit_rate": 0.0
            })
        }
        self.lock = threading.Lock()  # 线程锁
        self.clean_thread = threading.Thread(target=self._periodic_clean, daemon=True)
        self.clean_thread.start()
    
    def _generate_key(self, query: str, params: Dict) -> str:
        """生成缓存键（增加序列化容错）
"""
        try:
            # 过滤不可序列化的参数值
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, (dict, list, tuple)):
                    safe_params[k] = json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
                else:
                    safe_params[k] = str(v)
            
            key_components = [query, json.dumps(safe_params, sort_keys=True)]
            key_str = "_".join(key_components)
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"生成缓存键失败，使用降级策略: {e}")
            return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()
    
    def get(self, query: str, params: Dict) -> Optional[Any]:
        """线程安全获取缓存
"""
        with self.lock:
            self.stats["total_requests"] += 1
            key = self._generate_key(query, params)
            query_type = self._classify_query(query)

            if key in self.cache:
                cache_entry = self.cache[key]
                current_time = time.time()
                
                if current_time
"""线程安全设置缓存
"""with self.lock:
            key = self._generate_key(query, params)
            current_time = time.time()
            
            cache_entry = {
                "result": result,
                "timestamp": current_time,
                "query": query[:100]
            }
            
            if key in self.cache:
                del self.cache[key]
            else:
                self.stats["writes"] += 1
            
            self.cache[key] = cache_entry
            
            # 检查缓存大小
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats["evictions"] += 1
            
            # 更新统计
            self.stats["cache_size"] = len(self.cache)
            if self.stats["cache_size"] > self.stats["max_cache_size"]:
                self.stats["max_cache_size"] = self.stats["cache_size"]
    
    def set_vector_cache(self, query: str, vector: np.ndarray) -> None:
        """缓存查询向量
"""
        with self.lock:
            key = hashlib.md5(query.encode()).hexdigest()
            self.vector_cache[key] = {
                "vector": vector,
                "timestamp": time.time()
            }
    
    def get_vector_cache(self, query: str) -> Optional[np.ndarray]:
        """获取缓存的查询向量
"""
        with self.lock:
            key = hashlib.md5(query.encode()).hexdigest()
            if key in self.vector_cache:
                entry = self.vector_cache[key]
                if time.time()
"""获取缓存统计（返回副本避免篡改）
"""with self.lock:
            stats_copy = self.stats.copy()
            stats_copy["query_type_stats"] = {k: v.copy() for k, v in stats_copy["query_type_stats"].items()}
            
            # 计算命中率
            total = stats_copy["total_requests"]
            stats_copy["hit_rate"] = stats_copy["hits"] / total if total > 0 else 0.0
            
            # 计算各查询类型命中率
            for qtype, stats in stats_copy["query_type_stats"].items():
                total_q = stats["hits"] + stats["misses"]
                stats["hit_rate"] = stats["hits"] / total_q if total_q > 0 else 0.0
                stats["requests"] = total_q
            
            stats_copy["uptime"] = time.time()
"""清空缓存
"""with self.lock:
            self.cache.clear()
            self.vector_cache.clear()
            self.stats["cache_size"] = 0
            self.stats["last_clear_time"] = time.time()
    
    def remove_expired(self) -> None:
        """移除所有过期缓存
"""
        with self.lock:
            current_time = time.time()
            expired_keys = [k for k, v in self.cache.items() if current_time
"""定时清理过期缓存（后台线程）
"""while True:
            time.sleep(CONFIG["cache_clean_interval"])
            try:
                self.remove_expired()
                logger.debug("定时清理过期缓存完成")
            except Exception as e:
                logger.error(f"定时清理缓存失败: {e}")
    
    def _classify_query(self, query: str) -> str:
        """分类查询类型
"""
        if re.search(r'91[0-9A-Za-z]{16}', query):
            return "credit_code"
        elif self._is_address_query(query):
            return "address"
        elif self._is_business_scope_query(query):
            return "business_scope"
        else:
            return "general"
    
    @staticmethod
    def _is_address_query(query_text: str) -> bool:
        """补全：判断是否为地址查询（完整实现）
"""
        address_keywords = ['地址', '附近', '位于', '在', '注册地', '所在地', '坐落', '位置', '地址是', '周边', '地区', '区域']
        location_keywords = ['市', '区', '县', '镇', '街道', '路', '巷', '弄', '号', '村', '乡', '大道', '街', '园区', '工业区', '新区']
        region_keywords = ['中国', '省', '自治区', '直辖市', '特别行政区']
        
        has_address_keyword = any(kw in query_text for kw in address_keywords)
        has_location_keyword = any(kw in query_text for kw in location_keywords)
        has_region_keyword = any(kw in query_text for kw in region_keywords)
        has_numbered_address = bool(re.search(r'[路街道路巷弄号]+\d+', query_text))
        has_multi_location = len(re.findall(r'[市区县镇村街道]', query_text)) >= 2
        
        return (has_address_keyword and (has_location_keyword or has_region_keyword)) or has_numbered_address or has_multi_location
    
    @staticmethod
    def _is_business_scope_query(query_text: str) -> bool:
        """新增：判断是否为经营范围查询
"""
        business_keywords = [
            '经营范围', '经营项目', '主营', '兼营', '业务范围', 
            '许可项目', '一般项目', '经营许可', '资质', '营业范围'
        ]
        return any(kw in query_text for kw in business_keywords)

class OptimizedVectorDBQuery:
    """优化后的向量数据库查询器
"""
    
    def __init__(self, 
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = CONFIG["device"]):
        
        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"初始化向量数据库查询器 | 设备: {self.device}")
        
        # 初始化组件
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_cache = ThreadSafeQueryCache()
        
        # 配置（解耦硬编码）
        self.config = {
            "default_top_k": CONFIG["default_top_k"],
            "enable_cache": True,
            "enable_hybrid_search": True,
            "min_similarity_default": CONFIG["min_similarity_default"]
        }
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化组件（增强资源清理）
"""
        embedding_model = None
        chroma_client = None
        
        try:
            # 1. 加载Embedding模型
            logger.info("加载Embedding模型...")
            embedding_model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True
            )
            
            # 2. 初始化ChromaDB
            logger.info("连接ChromaDB...")
            chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取/创建集合
            try:
                collection = chroma_client.get_collection(name=self.collection_name)
            except Exception:
                collection = chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            # 初始化成功，赋值给实例变量
            self.embedding_model = embedding_model
            self.chroma_client = chroma_client
            self.collection = collection
            
            logger.info("✅ 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            # 清理已创建的资源
            if embedding_model and hasattr(embedding_model, 'cpu'):
                embedding_model.cpu()
                del embedding_model
            if chroma_client:
                chroma_client.close()
            raise
    
    def _validate_params(self, query_text: str, top_k: int, min_similarity: float) -> None:
        """参数校验（新增）
"""
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise ValueError("查询文本不能为空")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k必须为正整数（当前: {top_k}）")
        if min_similarity is not None and (min_similarity < 0 or min_similarity > 1):
            raise ValueError(f"min_similarity必须在0-1之间（当前: {min_similarity}）")
    
    def query(self, 
              query_text: str, 
              top_k: int = None,
              where_filter: Optional[Dict[str, Any]] = None,
              min_similarity: Optional[float] = None,
              return_format: str = "structured") -> Union[QueryResult, List[Dict[str, Any]]]:
        """核心查询方法（优化逻辑）
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            return_format: （structured/list）
"""
        start_time = time.time()
        
        # 参数默认值 + 校验
        top_k = top_k or self.config["default_top_k"]
        min_similarity = min_similarity or self.config["min_similarity_default"]
        
        try:
            self._validate_params(query_text, top_k, min_similarity)
        except ValueError as e:
            logger.error(f"参数校验失败: {e}")
            empty_result = QueryResult(
                query=query_text,
                retrieved_documents=[],
                retrieval_time=time.time()
"""优化：编码查询文本（带缓存）
"""
        # 优先从缓存获取
        cached_vector = self.query_cache.get_vector_cache(query_text)
        if cached_vector is not None:
            return cached_vector
        
        # 重新编码
        vector = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # 缓存向量
        if self.config["enable_cache"]:
            self.query_cache.set_vector_cache(query_text, vector)
        
        return vector
    
    def _vector_search_optimized(self, query_text: str, top_k: int, 
                           where_filter: Optional[Dict] = None,
                           min_similarity: Optional[float] = None) -> List[Dict]:
        """向量检索（优化空值处理和兼容性）
"""
        try:
            logger.info(f"向量检索 | 查询: {query_text[:100]}... | top_k: {top_k}")
            
            # 向量化（带缓存）
            vector_start = time.time()
            query_embedding = self._encode_query(query_text)
            logger.info(f"向量化完成 | 耗时: {time.time()-vector_start:.3f}s | 维度: {len(query_embedding)}")
            
            # 构建查询参数（安全校验）
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": max(1, top_k),  # 确保n_results≥1
                "include": ["documents", "metadatas", "distances", "ids"]
            }
            
            if where_filter and isinstance(where_filter, dict):
                # 校验过滤条件合法性（基础）
                valid_keys = ["$eq", "$ne", "$gt", "$lt", "$gte", "$lte", "$in", "$nin"]
                safe_where = {}
                for k, v in where_filter.items():
                    if isinstance(v, dict) and any(sk in v for sk in valid_keys):
                        safe_where[k] = v
                    else:
                        safe_where[k] = v
                query_params["where"] = safe_where
            
            # 执行查询
            query_start = time.time()
            results = self.collection.query(**query_params)
            logger.info(f"ChromaDB查询完成 | 耗时: {time.time()-query_start:.3f}s")
            
            # 处理结果（空值安全）
            retrieved_docs = []
            documents = results.get('documents', [[]])[0] or []
            metadatas = results.get('metadatas', [[]])[0] or [{}]*len(documents)
            distances = results.get('distances', [[]])[0] or [1.0]*len(documents)
            ids = results.get('ids', [[]])[0] or [f"doc_{i}" for i in range(len(documents))]
            
            total_docs = len(documents)
            filtered = 0
            
            for i, (doc, meta, dist, doc_id) in enumerate(zip(documents, metadatas, distances, ids)):
                if not doc:
                    continue
                
                similarity = 1
"""备选向量检索（增强容错）
"""try:
            query_embedding = self._encode_query(query_text)
            include_options = [
                ["documents", "metadatas", "distances"],
                ["documents", "distances"],
                ["documents", "metadatas"],
                ["documents"]
            ]
            
            for include in include_options:
                try:
                    query_params = {
                        "query_embeddings": [query_embedding.tolist()],
                        "n_results": max(1, top_k),
                        "include": include
                    }
                    if where_filter and isinstance(where_filter, dict):
                        query_params["where"] = where_filter
                    
                    results = self.collection.query(**query_params)
                    docs = []
                    documents = results.get('documents', [[]])[0] or []
                    
                    for i, doc in enumerate(documents):
                        if not doc:
                            continue
                        
                        # 相似度计算（降级）
                        dist = results.get('distances', [[]])[0][i] if (results.get('distances') and len(results['distances'][0])>i) else 0.5
                        similarity = 1
"""混合检索（优化权重逻辑和除零保护）
"""try:
            # 1. 获取向量检索候选结果
            candidate_top_k = top_k * CONFIG["hybrid_search_candidate_multiple"]
            vector_results = self._vector_search_optimized(query_text, candidate_top_k, where_filter, 0.0)
            
            if not vector_results:
                logger.warning("混合检索：无向量候选结果")
                return []
            
            # 2. 关键词提取（优化）
            tfidf_keywords = jieba.analyse.extract_tags(query_text, topK=15, withWeight=True)
            textrank_keywords = jieba.analyse.textrank(query_text, topK=10, withWeight=True)
            
            all_keywords = defaultdict(float)
            for word, weight in tfidf_keywords + textrank_keywords:
                all_keywords[word] += weight
            keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
            keyword_count = len(keywords)
            
            # 3. 混合得分计算
            hybrid_docs = []
            query_length = len(query_text.strip())
            
            for doc in vector_results:
                vector_score = doc.get('similarity', 0.0)
                content = doc.get('content', '').lower()
                keyword_score = 0.0
                matched_keywords = []
                
                if content and keyword_count > 0:
                    # 关键词匹配
                    for word, weight in keywords:
                        if word.lower() in content:
                            matched_keywords.append(weight)
                    
                    # 关键词得分（除零保护）
                    if matched_keywords:
                        base_score = sum(matched_keywords) / len(matched_keywords)
                        coverage = len(matched_keywords) / keyword_count
                        keyword_score = base_score * (1 + coverage * CONFIG["coverage_bonus_factor"])
                        
                        # 关键词位置增强
                        for word, _ in keywords[:5]:
                            if word.lower() in content[:200]:
                                keyword_score += CONFIG["keyword_position_bonus"]
                                break
                
                # 动态权重（解耦配置）
                if query_length < CONFIG["short_query_threshold"] and keyword_count < CONFIG["short_keyword_threshold"]:
                    v_weight = CONFIG["short_query_vector_weight"]
                    k_weight = CONFIG["short_query_keyword_weight"]
                elif query_length > CONFIG["long_query_threshold"] or keyword_count > CONFIG["long_keyword_threshold"]:
                    v_weight = CONFIG["long_query_vector_weight"]
                    k_weight = CONFIG["long_query_keyword_weight"]
                else:
                    v_weight = CONFIG["medium_query_vector_weight"]
                    k_weight = CONFIG["medium_query_keyword_weight"]
                
                # 混合得分
                hybrid_score = (vector_score * v_weight) + (keyword_score * k_weight)
                
                # 额外增强（边界保护）
                if query_text.lower() in content and hybrid_score + CONFIG["exact_query_match_bonus"] <= 1.0:
                    hybrid_score += CONFIG["exact_query_match_bonus"]
                if len(matched_keywords) > 5 and hybrid_score + CONFIG["multi_keyword_match_bonus"] <= 1.0:
                    hybrid_score += CONFIG["multi_keyword_match_bonus"]
                
                # 结果封装
                doc_copy = doc.copy()
                doc_copy.update({
                    "similarity": round(min(hybrid_score, 1.0), 4),  # 限制最大得分1.0
                    "vector_score": round(vector_score, 4),
                    "keyword_score": round(keyword_score, 4),
                    "vector_weight": v_weight,
                    "keyword_weight": k_weight
                })
                hybrid_docs.append(doc_copy)
            
            # 4. 过滤+排序+重排
            if min_similarity is not None:
                hybrid_docs = [d for d in hybrid_docs if d["similarity"] >= min_similarity]
            
            hybrid_docs.sort(key=lambda x: x["similarity"], reverse=True)
            hybrid_docs = self._rerank_results(query_text, hybrid_docs, top_k)
            
            return hybrid_docs[:top_k]
            
        except Exception as e:
            logger.error(f"混合检索失败，返回向量检索结果: {e}")
            return self._vector_search_optimized(query_text, top_k, where_filter, min_similarity)
    
    def _credit_code_exact_match(self, query_text: str, top_k: int,
                                where_filter: Optional[Dict] = None,
                                min_similarity: Optional[float] = None) -> List[Dict]:
        """信用代码精确匹配（重构逻辑，优先精确查询）
"""
        try:
            # 提取信用代码
            credit_code_pattern = r'91[0-9A-Z]{16}'
            match = re.search(credit_code_pattern, query_text.upper())
            
            if not match:
                logger.info("未提取到信用代码，执行混合检索")
                return self._hybrid_search(query_text, top_k, where_filter, min_similarity)
            
            credit_code = match.group(0)
            logger.info(f"提取到信用代码: {credit_code}")
            
            # 优化：优先通过元数据精确查询（向量检索本末倒置问题修复）
            exact_where = {"credit_code": {"$eq": credit_code}} if where_filter is None else {**where_filter, **{"credit_code": {"$eq": credit_code}}}
            exact_results = self._vector_search_optimized(credit_code, top_k, exact_where, 0.0)
            
            # 精确匹配筛选
            exact_matched = []
            for doc in exact_results:
                content = doc.get('content', '').upper()
                doc_id = doc.get('id', '').upper()
                
                if credit_code in content:
                    doc["similarity"] = CONFIG["credit_code_exact_match_score"]
                    doc["exact_match"] = True
                    exact_matched.append(doc)
                elif credit_code in doc_id:
                    doc["similarity"] = CONFIG["credit_code_id_match_score"]
                    doc["exact_match"] = True
                    exact_matched.append(doc)
            
            if exact_matched:
                logger.info(f"找到 {len(exact_matched)} 个精确匹配结果")
                return exact_matched[:top_k]
            
            # 无精确匹配，扩大检索范围
            logger.warning("无精确匹配，扩大候选范围")
            candidate_results = self._vector_search_optimized(
                credit_code, 
                top_k * CONFIG["credit_code_candidate_multiple"],
                where_filter,
                0.0
            )
            
            return candidate_results[:top_k]
            
        except Exception as e:
            logger.error(f"信用代码匹配失败: {e}")
            return self._hybrid_search(query_text, top_k, where_filter, min_similarity)
    
    def _address_optimized_search(self, query_text: str, top_k: int,
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        """新增：地址优化检索
"""
        try:
            logger.info(f"地址优化检索 | 查询: {query_text[:100]}...")
            
            # 改进的地址规范化处理
            def normalize_address(address: str) -> str:
                """规范化地址格式，支持更多地址表述
"""
                # 移除空格和特殊字符
                normalized = re.sub(r'[^\u4e00-\u9fa50-9]', '', address)
                # 统一地址术语
                normalized = normalized.replace('大道', '路')
                normalized = normalized.replace('大街', '街')
                normalized = normalized.replace('工业区', '园区')
                normalized = normalized.replace('新区', '区')
                normalized = normalized.replace('旅游区', '区')
                normalized = normalized.replace('经济开发区', '开发区')
                normalized = normalized.replace('高新技术产业开发区', '高新区')
                normalized = normalized.replace('保税区', '区')
                normalized = normalized.replace('出口加工区', '加工区')
                normalized = normalized.replace('科技园', '园区')
                normalized = normalized.replace('创业园', '园区')
                normalized = normalized.replace('产业园', '园区')
                # 统一门牌号表述
                normalized = normalized.replace('号院', '号')
                normalized = normalized.replace('号附', '号')
                normalized = normalized.replace('号之', '号')
                return normalized
            
            # 提取地址关键词
"""提取地址关键词，包括完整地址和部分地址
"""keywords = []
                
                # 1. 提取完整地址
                full_address_patterns = [
                    r'([\u4e00-\u9fa5]+[省市])([\u4e00-\u9fa5]+[县区])([\u4e00-\u9fa5]+[乡镇街道])([\u4e00-\u9fa5]+[路街道路巷弄])(\d+[号院附之]*)',
                    r'([\u4e00-\u9fa5]+[省市县区])([\u4e00-\u9fa5]+[乡镇街道])([\u4e00-\u9fa5]+[路街道路巷弄])(\d+[号院附之]*)',
                    r'([\u4e00-\u9fa5]+[乡镇街道])([\u4e00-\u9fa5]+[路街道路巷弄])(\d+[号院附之]*)',
                    r'([\u4e00-\u9fa5]+[路街道路巷弄])(\d+[号院附之]*)',
                    r'(\d+[号院附之]*)([\u4e00-\u9fa5]+[路街道路巷弄])'
                ]
                
                for pattern in full_address_patterns:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if isinstance(match, tuple):
                            # 提取所有非空匹配组
                            parts = [m for m in match if m]
                            # 添加完整地址
                            if parts:
                                full_address = ''.join(parts)
                                keywords.append(full_address)
                                # 添加各个部分
                                keywords.extend(parts)
                        else:
                            keywords.append(match)
                
                # 2. 提取行政区域
                region_patterns = [
                    r'([\u4e00-\u9fa5]+[省市县区乡镇街道])',
                    r'([\u4e00-\u9fa5]+[村社区])'
                ]
                
                for pattern in region_patterns:
                    matches = re.findall(pattern, text)
                    keywords.extend(matches)
                
                # 3. 去重并过滤空关键词
                keywords = list(set(filter(None, keywords)))
                
                # 4. 按长度排序，优先匹配长关键词
                keywords.sort(key=lambda x: len(x), reverse=True)
                
                return keywords
            
            # 提取地址关键词
            address_keywords = extract_address_keywords(query_text)
            logger.debug(f"提取到地址关键词: {address_keywords}")
            
            # 获取更多候选结果
            hybrid_results = self._hybrid_search(
                query_text,
                top_k * 15,  # 进一步增加候选数量
                where_filter,
                0.0  # 初始检索不设置相似度阈值
            )
            
            if not hybrid_results:
                logger.warning("未获取到混合检索结果")
                return []
            
            # 地址相关性重排
"""新增：经营范围增强检索
"""try:
            logger.info(f"经营范围检索 | 查询: {query_text[:100]}...")
            
            # 优化1: 更精确的关键词提取
            def extract_business_keywords(text: str) -> List[Tuple[str, float]]:
                """提取经营范围关键词，返回带权重的关键词列表
"""
                # 移除经营范围相关术语
                text = re.sub(r'经营范围|经营项目|主营|兼营|业务范围|许可项目|一般项目|经营许可|资质|营业范围|依法须经批准的项目|经相关部门批准后方可开展经营活动|具体经营项目以审批结果为准', '', text)
                
                # 提取关键词
                tfidf_keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True)
                textrank_keywords = jieba.analyse.textrank(text, topK=15, withWeight=True)
                
                # 合并关键词并加权
                all_keywords = defaultdict(float)
                for word, weight in tfidf_keywords:
                    all_keywords[word] += weight * 1.0  # TF-IDF权重
                for word, weight in textrank_keywords:
                    all_keywords[word] += weight * 0.8  # TextRank权重，稍微降低
                
                # 添加行业相关术语作为额外关键词
                industry_terms = {
                    '建筑': 0.5,
                    '装饰': 0.5,
                    '装修': 0.5,
                    '工程': 0.4,
                    '设计': 0.4,
                    '施工': 0.4,
                    '建材': 0.4,
                    '贸易': 0.3,
                    '销售': 0.3,
                    '批发': 0.3,
                    '零售': 0.3,
                    '科技': 0.3,
                    '技术': 0.3,
                    '服务': 0.3,
                    '咨询': 0.3,
                    '开发': 0.3,
                    '制造': 0.3,
                    '加工': 0.3
                }
                
                for term, weight in industry_terms.items():
                    if term in text:
                        all_keywords[term] += weight
                
                # 排序并返回前25个关键词
                keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:25]
                return keywords
            
            # 提取关键词
            business_keywords = extract_business_keywords(query_text)
            logger.debug(f"提取到经营范围关键词: {[word for word, weight in business_keywords]}")
            
            # 优化2: 获取更多候选结果
            hybrid_results = self._hybrid_search(
                query_text,
                top_k * 20,  # 进一步增加候选数量
                where_filter,
                0.0  # 初始检索不设置相似度阈值
            )
            
            if not hybrid_results:
                logger.warning("未获取到混合检索结果")
                return []
            
            # 优化3: 经营范围相关性评分
"""计算经营范围匹配得分
"""
                if not query_keywords:
                    return 0.0
                
                # 内容预处理
                content = content.lower()
                
                # 关键词匹配情况
                matched_keywords = []
                total_weight = 0.0
                matched_weight = 0.0
                
                # 1. 带权重的关键词匹配
                for keyword, weight in query_keywords:
                    if keyword in content:
                        matched_keywords.append(keyword)
                        matched_weight += weight
                        total_weight += weight
                    else:
                        total_weight += weight
                
                if not matched_keywords:
                    return 0.0
                
                # 2. 匹配覆盖率（带权重）
                coverage_score = matched_weight / total_weight if total_weight > 0 else 0.0
                
                # 3. 匹配密度（匹配关键词数/内容长度）
                density_score = len(matched_keywords) / len(content) * 100
                
                # 4. 完整业务短语匹配
                phrase_match_bonus = 0.0
                # 提取查询中的业务短语
                business_phrases = re.findall(r'([\u4e00-\u9fa5]{2,5}[行业业务服务项目])', query_text)
                for phrase in business_phrases:
                    if phrase in content:
                        phrase_match_bonus += 0.2
                
                # 5. 核心业务关键词匹配
                core_keywords = [word for word, weight in query_keywords[:5]]  # 前5个核心关键词
                core_matched = sum(1 for keyword in core_keywords if keyword in content)
                core_match_score = core_matched / len(core_keywords) if core_keywords else 0.0
                
                # 6. 行业匹配度
                industry_match_bonus = 0.0
                industry_keywords = ['建筑', '装饰', '装修', '工程', '设计', '施工', '建材', '贸易', '销售', '科技', '技术', '服务', '咨询', '开发', '制造', '加工']
                query_industries = [kw for kw in industry_keywords if kw in query_text]
                if query_industries:
                    for industry in query_industries:
                        if industry in content:
                            industry_match_bonus += 0.15
                
                # 综合得分
                total_score = (
                    coverage_score * 0.4 +
                    density_score * 0.15 +
                    phrase_match_bonus * 0.15 +
                    core_match_score * 0.15 +
                    industry_match_bonus * 0.15
                )
                
                return min(total_score, 1.0)  # 限制最大得分
            
            # 优化4: 结果重排
            business_reranked = []
            
            for doc in hybrid_results:
                content = doc.get('content', '')
                vector_score = doc.get('similarity', 0.0)
                
                # 计算经营范围匹配得分
                business_score = calculate_business_scope_score(business_keywords, content)
                
                # 提取元数据中的经营范围信息（如果有）
                metadata = doc.get('metadata', {})
                business_scope_metadata = metadata.get('business_scope', '')
                
                # 元数据经营范围匹配加分
                metadata_bonus = 0.0
                if business_scope_metadata:
                    metadata_content = business_scope_metadata.lower()
                    # 检查核心关键词是否在元数据中
                    core_keywords = [word for word, weight in business_keywords[:5]]
                    core_matched_in_meta = sum(1 for keyword in core_keywords if keyword in metadata_content)
                    if core_matched_in_meta > 0:
                        metadata_bonus += core_matched_in_meta * 0.1
                
                # 融合得分：向量得分(0.3) + 经营范围得分(0.7) + 元数据奖励
                total_score = (vector_score * 0.3) + (business_score * 0.7) + metadata_bonus
                total_score = min(total_score, 1.0)
                
                doc_copy = doc.copy()
                doc_copy.update({
                    "similarity": round(total_score, 4),
                    "vector_score": round(vector_score, 4),
                    "business_score": round(business_score, 4),
                    "metadata_bonus": round(metadata_bonus, 4),
                    "matched_business_keywords": len([kw for kw, weight in business_keywords if kw in content]),
                    "total_business_keywords": len(business_keywords),
                    "core_keywords_matched": len([word for word, weight in business_keywords[:5] if word in content])
                })
                
                business_reranked.append(doc_copy)
            
            # 按相似度排序
            business_reranked.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 应用最终相似度阈值
            if min_similarity is not None:
                business_reranked = [doc for doc in business_reranked if doc["similarity"] >= min_similarity]
            
            return business_reranked[:top_k]
            
        except Exception as e:
            logger.error(f"经营范围检索失败: {e}")
            return self._hybrid_search(query_text, top_k, where_filter, min_similarity)
    
    def _rerank_results(self, query_text: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """新增：结果重排（基础版）
"""
        try:
            # 简单BM25重排
            if not docs:
                return []
            
            # 提取文档内容
            corpus = [doc.get('content', '') for doc in docs]
            tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
            tokenized_query = jieba.lcut(query_text)
            
            # BM25计算
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # 融合BM25得分
            reranked_docs = []
            for doc, bm25_score in zip(docs, bm25_scores):
                # 归一化BM25得分
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
                normalized_bm25 = bm25_score / max_bm25
                
                # 融合得分
                doc["similarity"] = round((doc["similarity"] * 0.7) + (normalized_bm25 * 0.3), 4)
                doc["reranked"] = True
                reranked_docs.append(doc)
            
            # 重新排序
            reranked_docs.sort(key=lambda x: x["similarity"], reverse=True)
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.warning(f"结果重排失败，返回原结果: {e}")
            return docs[:top_k]
    
    @staticmethod
    def _is_credit_code(self, query_text: str) -> bool:
        """信用代码判断（优化正则）
"""
        credit_code_pattern = r'91[0-9a-zA-Z]{16}'
        credit_code_keywords = ['统一社会信用代码', '统一信用代码', '信用代码', '社会信用代码', '纳税人识别号']
        
        has_credit_code = bool(re.search(credit_code_pattern, query_text, re.IGNORECASE))
        has_credit_keyword = any(kw in query_text for kw in credit_code_keywords)
        
        return has_credit_code or has_credit_keyword
    
    def _is_company_name_query(self, query_text: str) -> bool:
        """判断是否为公司名称查询
"""
        # 公司名称关键词
        company_keywords = ['有限公司', '有限责任公司', '股份有限公司', '分公司', '子公司', '总公司', '集团公司', '企业', '公司']
        
        # 检查是否包含公司名称关键词
        has_company_keyword = any(kw in query_text for kw in company_keywords)
        
        # 检查是否为公司名称查询模式
        company_name_patterns = [
            r'[\u4e00-\u9fa5]+有限公司',
            r'[\u4e00-\u9fa5]+有限责任公司',
            r'[\u4e00-\u9fa5]+股份有限公司',
            r'[\u4e00-\u9fa5]+分公司',
            r'[\u4e00-\u9fa5]+子公司',
            r'[\u4e00-\u9fa5]+集团公司'
        ]
        
        has_company_pattern = any(re.search(pattern, query_text) for pattern in company_name_patterns)
        
        return has_company_keyword or has_company_pattern
    
    def test_connection(self) -> Dict[str, Any]:
        """测试模型和数据库连接是否正常
        :return: 连接测试结果（包含状态和错误信息）
"""
        test_result = {
            "model_loaded": False,
            "db_connected": False,
            "collection_exists": False,
            "test_query_success": False,
            "error_msg": "",
            "collection_count": 0
        }
        
        try:
            # 1. 验证Embedding模型是否加载正常
            if self.embedding_model is not None:
                # 测试模型编码功能
                test_text = "测试连接"
                self._encode_query(test_text)  # 复用已有的编码方法
                test_result["model_loaded"] = True
            else:
                test_result["error_msg"] += "Embedding模型未加载; "
            
            # 2. 验证ChromaDB连接是否正常
            if self.chroma_client is not None:
                test_result["db_connected"] = True
                
                # 3. 验证集合是否存在
                try:
                    # 获取集合信息
                    self.collection.get()  # 验证集合可访问
                    test_result["collection_exists"] = True
                    
                    # 4. 获取集合文档数量（可选）
                    count = self.collection.count()
                    test_result["collection_count"] = count
                except Exception as e:
                    test_result["error_msg"] += f"集合访问失败: {str(e)}; "
            
            # 5. 执行简单测试查询（验证端到端功能）
            try:
                test_query_result = self.query(
                    query_text="测试连接",
                    top_k=1,
                    min_similarity=0.0
                )
                test_result["test_query_success"] = True
            except Exception as e:
                test_result["error_msg"] += f"测试查询失败: {str(e)}; "
            
            # 清理错误信息末尾的多余符号
            test_result["error_msg"] = test_result["error_msg"].rstrip("; ").strip()
            return test_result
            
        except Exception as e:
            test_result["error_msg"] = f"连接测试异常: {str(e)}"
            logger.error(f"连接测试失败: {e}")
            return test_result
    
    def close(self) -> None:
        """新增：资源释放
"""
        logger.info("释放资源...")
        # 释放模型资源
        if self.embedding_model:
            self.embedding_model.cpu()
            del self.embedding_model
        # 关闭ChromaDB连接
        if self.chroma_client:
            self.chroma_client.close()
        # 清空缓存
        self.query_cache.clear()
        logger.info("资源释放完成")

# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 初始化查询器
    queryer = OptimizedVectorDBQuery(
        db_path="./chroma_db",
        model_path="./bge-m3",
        collection_name="test_collection"
    )
    
    try:
        # 执行查询
        result = queryer.query(
            query_text="北京市海淀区中关村大街1号 统一信用代码91110108MA00000000",
            top_k=10,
            min_similarity=0.5
        )
        
        # 打印结果
        print(f"查询耗时: {result.retrieval_time:.3f}s")
        print(f"返回结果数: {result.total_retrieved}")
        print(f"平均相似度: {result.avg_similarity:.4f}")
        
        # 打印缓存统计
        stats = queryer.query_cache.get_stats()
        print(f"缓存命中率: {stats['hit_rate']:.2%}")
        
    finally:
        # 释放资源
        queryer.close()