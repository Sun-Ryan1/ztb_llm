#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询器
"""

import os
import json
import time
import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass, field
from datetime import datetime
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
        """转换为字典格式
"""
        return {
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "retrieval_time": self.retrieval_time,
            "avg_similarity": self.avg_similarity,
            "query_id": self.query_id,
            "retrieval_method": self.retrieval_method,
            "reranked": self.reranked,
            "retrieved_documents": self.retrieved_documents
        }

class QueryCache:
    """查询缓存系统
"""
    
    def __init__(self, max_size: int = 1000, expire_time: int = 3600):
        """初始化查询缓存
        
        Args:
            _size: 
            expire_time: （秒）
"""
        self.max_size = max_size
        self.expire_time = expire_time
        self.cache = OrderedDict()
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
            "query_type_stats": defaultdict(lambda: {
                "hits": 0,
                "misses": 0,
                "requests": 0
            })
        }
    
    def _generate_key(self, query: str, params: Dict) -> str:
        """生成缓存键
"""
        # 包含更多参数，确保缓存的唯一性
        key_components = [
            query,
            json.dumps(params, sort_keys=True, ensure_ascii=False)
        ]
        key_str = "_".join(key_components)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, params: Dict) -> Optional[Any]:
        """获取缓存结果
"""
        self.stats["total_requests"] += 1
        key = self._generate_key(query, params)
        
        # 提取查询类型用于统计
        query_type = self._classify_query(query)
        
        if key in self.cache:
            # 检查缓存是否过期
            cache_entry = self.cache[key]
            current_time = time.time()
            
            if current_time
"""设置缓存结果
"""key = self._generate_key(query, params)
        current_time = time.time()
        
        # 构建缓存条目
        cache_entry = {
            "result": result,
            "timestamp": current_time,
            "query": query[:100]  # 保存查询前缀用于调试
        }
        
        if key in self.cache:
            # 更新现有缓存
            del self.cache[key]
        else:
            self.stats["writes"] += 1
        
        # 添加到缓存末尾
        self.cache[key] = cache_entry
        
        # 检查缓存大小
        if len(self.cache) > self.max_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
        
        # 更新统计
        self.stats["cache_size"] = len(self.cache)
        # 更新最大缓存大小
        if self.stats["cache_size"] > self.stats["max_cache_size"]:
            self.stats["max_cache_size"] = self.stats["cache_size"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
"""
        # 计算命中率
        if self.stats["total_requests"] > 0:
            self.stats["hit_rate"] = self.stats["hits"] / self.stats["total_requests"]
        else:
            self.stats["hit_rate"] = 0.0
        
        # 更新各查询类型的命中率
        for qtype, stats in self.stats["query_type_stats"].items():
            total = stats["hits"] + stats["misses"]
            if total > 0:
                stats["hit_rate"] = stats["hits"] / total
            else:
                stats["hit_rate"] = 0.0
            stats["requests"] = total
        
        return {
            **self.stats,
            "uptime": time.time()
"""清空缓存
"""self.cache.clear()
        self.stats["cache_size"] = 0
        self.stats["last_clear_time"] = time.time()
    
    def remove_expired(self) -> None:
        """移除所有过期缓存
"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time
"""简单分类查询类型用于统计
"""
        # 信用代码查询
        if re.search(r'91[0-9A-Za-z]{16}', query):
            return "credit_code"
        # 地址查询
        elif self._is_address_query(query):
            return "address"
        # 经营范围查询
        elif self._is_business_scope_query(query):
            return "business_scope"
        # 其他查询
        else:
            return "general"

class OptimizedVectorDBQuery:
    """向量数据库查询器
"""
    
    def __init__(self, 
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = None):
        
        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        
        # 自动选择设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("初始化版向量数据库查询器")
        logger.info(f"设备: {self.device}")
        
        # 初始化组件
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_cache = QueryCache(max_size=1000, expire_time=3600)  # 增加缓存大小
        
        # 配置
        self.config = {
            "default_top_k": 20,  # 增加默认返回结果数量到20
            "enable_cache": True,
            "enable_hybrid_search": True,  # 启用混合检索
        }
        
        # 初始化所有组件
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化所有组件
"""
        try:
            # 1. 初始化embedding模型
            logger.info("加载Embedding模型...")
            self.embedding_model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True
            )
            
            # 2. 初始化ChromaDB
            logger.info("连接ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取集合（如果不存在会创建）
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except:
                # 如果集合不存在，创建新集合
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info("✅ 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def query(self, 
              query_text: str, 
              top_k: int = None,
              where_filter: Optional[Dict[str, Any]] = None,
              min_similarity: Optional[float] = 0.0,
              return_format: str = "structured") -> Union[QueryResult, List[Dict[str, Any]]]:
        """查询
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            return_format:
"""
        start_time = time.time()
        
        # 使用默认值
        if top_k is None:
            top_k = self.config["default_top_k"]
        
        # 检查缓存
        cache_params = {
            "query": query_text,
            "top_k": top_k,
            "where_filter": where_filter,
            "min_similarity": min_similarity
        }
        
        if self.config["enable_cache"]:
            cached_result = self.query_cache.get(query_text, cache_params)
            if cached_result:
                if return_format == "list":
                    return cached_result["retrieved_documents"]
                else:
                    return QueryResult(**cached_result)
        
        try:
            # 1. 检查是否为信用代码查询，优先使用精确匹配
            if self._is_credit_code(query_text):
                # 执行信用代码精确匹配
                retrieved_docs = self._credit_code_exact_match(query_text, top_k)
                retrieval_method = "credit_code_exact"
            # 2. 检查是否为地址查询，使用地址优化检索
            elif self._is_address_query(query_text):
                # 执行地址优化检索
                retrieved_docs = self._address_optimized_search(query_text, top_k, where_filter, min_similarity)
                retrieval_method = "address_optimized"
            # 3. 检查是否为经营范围查询，使用关键词增强检索
            elif self._is_business_scope_query(query_text):
                # 执行经营范围增强检索
                retrieved_docs = self._business_scope_enhanced_search(query_text, top_k, where_filter, min_similarity)
                retrieval_method = "business_scope_enhanced"
            # 4. 根据配置选择常规检索方式
            elif self.config["enable_hybrid_search"]:
                # 执行混合检索
                retrieved_docs = self._hybrid_search(
                    query_text, 
                    top_k=top_k,
                    where_filter=where_filter,
                    min_similarity=min_similarity
                )
                retrieval_method = "hybrid_search"
            else:
                # 执行向量检索
                retrieved_docs = self._vector_search_optimized(
                    query_text, 
                    top_k=top_k,
                    where_filter=where_filter,
                    min_similarity=min_similarity
                )
                retrieval_method = "vector_search"
            
            # 取top_k个结果
            final_docs = retrieved_docs[:top_k]
            
            # 计算统计信息
            retrieval_time = time.time()
"""向量检索（兼容新版ChromaDB）
"""try:
            logger.info(f"开始向量检索，查询文本: {query_text[:100]}..., top_k: {top_k}, where_filter: {where_filter}, min_similarity: {min_similarity}")
            
            # 向量化
            vector_start_time = time.time()
            query_embedding = self.embedding_model.encode(
                query_text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            vector_time = time.time()
"""备选向量检索方法
"""try:
            # 向量化
            query_embedding = self.embedding_model.encode(
                query_text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # 尝试不同的API调用方式
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            # 尝试不同的include参数组合
            include_options = [
                ["documents", "metadatas", "distances"],
                ["documents", "distances"],
                ["documents", "metadatas"],
                ["documents"]
            ]
            
            for include_option in include_options:
                try:
                    query_params["include"] = include_option
                    results = self.collection.query(**query_params)
                    
                    # 处理结果
                    retrieved_docs = []
                    if results['documents'] and results['documents'][0]:
                        for i, doc_content in enumerate(results['documents'][0]):
                            # 获取相似度
                            if 'distances' in results and results['distances'][0]:
                                distance = results['distances'][0][i]
                                similarity = 1
"""混合检索：结合向量检索和关键词检索
"""try:
            # 1. 执行向量检索，获取更多结果用于混合
            vector_results = self._vector_search_optimized(
                query_text, 
                top_k=top_k * 3,  # 增加候选结果数量
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            
            if not vector_results:
                return []
            
            # 2. 关键词提取优化
            # 提取不同类型的关键词
            # 2.1 基于TF-IDF的关键词
            tfidf_keywords = jieba.analyse.extract_tags(query_text, topK=15, withWeight=True)
            # 2.2 基于TextRank的关键词
            textrank_keywords = jieba.analyse.textrank(query_text, topK=10, withWeight=True)
            # 2.3 合并关键词，去重
            all_keywords = {}
            for word, weight in tfidf_keywords + textrank_keywords:
                if word in all_keywords:
                    all_keywords[word] += weight
                else:
                    all_keywords[word] = weight
            # 转换回列表格式
            keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # 3. 计算每个文档的混合得分
            hybrid_docs = []
            for doc in vector_results:
                # 向量相似度
                vector_score = doc.get('similarity', 0)
                
                # 关键词匹配得分
                content = doc.get('content', '')
                keyword_score = 0.0
                
                if content:
                    # 3.1 统计文档中包含的关键词数量和权重
                    matched_keywords = []
                    for word, weight in keywords:
                        if word in content:
                            matched_keywords.append(weight)
                    
                    if matched_keywords:
                        # 基础关键词得分
                        keyword_score = sum(matched_keywords) / len(matched_keywords)
                        
                        # 3.2 关键词覆盖率增强
                        coverage = len(matched_keywords) / len(keywords) if keywords else 0
                        keyword_score *= (1 + coverage * 0.5)  # 覆盖率越高，得分加成越多
                        
                        # 3.3 关键词位置增强
                        # 检查关键词是否出现在文档开头
                        content_lower = content.lower()
                        for word, _ in keywords[:5]:  # 只检查前5个重要关键词
                            if word.lower() in content_lower[:200]:  # 文档前200字符
                                keyword_score += 0.1  # 每个重要关键词出现在开头加0.1分
                                break
                
                # 4. 动态调整混合权重
                # 根据查询长度和关键词数量动态调整权重
                query_length = len(query_text)
                keyword_count = len(keywords)
                
                if query_length < 10 and keyword_count < 3:
                    # 短查询，依赖向量检索更多
                    vector_weight = 0.7
                    keyword_weight = 0.3
                elif query_length > 50 or keyword_count > 10:
                    # 长查询，关键词更重要
                    vector_weight = 0.5
                    keyword_weight = 0.5
                else:
                    # 中等长度查询，平衡权重
                    vector_weight = 0.6
                    keyword_weight = 0.4
                
                # 5. 计算混合得分
                hybrid_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
                
                # 6. 添加额外的相关性增强
                # 6.1 如果查询完全出现在文档中，增加得分
                if query_text in content:
                    hybrid_score += 0.2
                # 6.2 如果文档包含多个相同类型的关键词，增加得分
                if len(matched_keywords) > 5:
                    hybrid_score += 0.1
                
                # 更新文档的相似度为混合得分
                doc_copy = doc.copy()
                doc_copy['similarity'] = hybrid_score
                doc_copy['vector_score'] = vector_score
                doc_copy['keyword_score'] = keyword_score
                doc_copy['vector_weight'] = vector_weight
                doc_copy['keyword_weight'] = keyword_weight
                hybrid_docs.append(doc_copy)
            
            # 7. 按混合得分排序
            hybrid_docs.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 8. 过滤掉低相似度文档
            if min_similarity is not None:
                hybrid_docs = [doc for doc in hybrid_docs if doc['similarity'] >= min_similarity]
            
            # 9. 结果重排序
            hybrid_docs = self._rerank_results(query_text, hybrid_docs, top_k)
            
            return hybrid_docs
            
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            # 失败时返回向量检索结果
            return self._vector_search_optimized(
                query_text, 
                top_k=top_k,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
    
    def _is_credit_code(self, query_text: str) -> bool:
        """判断是否为信用代码查询
        
        Args:
            _text: 
            
        Returns:
            : 是否为信用代码查询
"""
        # 信用代码正则：18位字母数字，通常以91开头，匹配大小写
        credit_code_pattern = r'91[0-9a-zA-Z]{16}'
        # 同时检测查询中是否包含"统一信用代码"等关键词
        credit_code_keywords = ['统一社会信用代码', '统一信用代码', '信用代码', '社会信用代码', '纳税人识别号']
        
        has_credit_code = bool(re.search(credit_code_pattern, query_text, re.IGNORECASE))
        has_credit_keyword = any(keyword in query_text for keyword in credit_code_keywords)
        
        return has_credit_code or has_credit_keyword
    
    def _credit_code_exact_match(self, query_text: str, top_k: int) -> List[Dict]:
        """信用代码精确匹配
        
        Args:
            _text: 
            top_k: 
            
        Returns:
            List[Dict]: 匹配的文档列表
"""
        try:
            # 提取信用代码
            credit_code_pattern = r'91[0-9A-Z]{16}'
            match = re.search(credit_code_pattern, query_text.upper())
            
            if match:
                credit_code = match.group(0)
                logger.info(f"提取到信用代码: {credit_code}")
                
                # 1. 执行向量检索，获取相关结果
                vector_results = self._vector_search_optimized(
                    credit_code, 
                    top_k=top_k * 10,  # 获取更多结果用于筛选
                    min_similarity=0.0  # 不设置相似度阈值，确保获取所有可能结果
                )
                
                # 2. 执行混合检索，获取更多候选结果
                hybrid_results = self._hybrid_search(
                    credit_code, 
                    top_k=top_k * 10,
                    min_similarity=0.0
                )
                
                # 合并结果，去重
                all_results = []
                seen_ids = set()
                
                # 先添加向量检索结果
                for doc in vector_results:
                    if doc.get('id') not in seen_ids:
                        seen_ids.add(doc.get('id'))
                        all_results.append(doc)
                
                # 再添加混合检索结果
                for doc in hybrid_results:
                    if doc.get('id') not in seen_ids:
                        seen_ids.add(doc.get('id'))
                        all_results.append(doc)
                
                logger.info(f"获取到 {len(all_results)} 个候选结果")
                
                # 3. 精确匹配筛选
"""判断是否为地址查询
        
        Args:
            _text: 
            
        Returns:
            : 是否为地址查询
"""
        # 地址关键词
        address_keywords = ['地址', '附近', '位于', '在', '注册地', '所在地', '坐落', '位置', '地址是', '位于', '周边', '地区', '区域']
        # 地名关键词
        location_keywords = ['市', '区', '县', '镇', '街道', '路', '巷', '弄', '号', '村', '乡', '大道', '街', '园区', '工业区', '新区']
        # 区域关键词
        region_keywords = ['中国', '省', '自治区', '直辖市', '特别行政区']
        
        # 检查是否包含地址相关关键词
        has_address_keyword = any(keyword in query_text for keyword in address_keywords)
        has_location_keyword = any(keyword in query_text for keyword in location_keywords)
        has_region_keyword = any(keyword in query_text for keyword in region_keywords)
        
        # 检查是否包含数字地址（如XX路XX号）
        has_numbered_address = bool(re.search(r'[路街道路巷弄号]+\d+', query_text))
        
        # 检查是否包含多个地名关键词（如XX市XX区）
        location_count = sum(1 for keyword in location_keywords if keyword in query_text)
        has_multiple_locations = location_count >= 2
        
        # 包含地址关键词，或者包含地名关键词+区域关键词，或者包含数字地址，或者包含多个地名关键词
        return has_address_keyword or (has_location_keyword and has_region_keyword) or has_numbered_address or has_multiple_locations
    
    def _address_optimized_search(self, query_text: str, top_k: int, 
                                 where_filter: Optional[Dict] = None,
                                 min_similarity: Optional[float] = None) -> List[Dict]:
        """地址优化检索
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            
        Returns:
            List[Dict]: 检索结果
"""
        try:
            logger.info(f"开始地址优化检索: {query_text}")
            
            # 提取地址相关信息
            # 1. 提取完整地址模式
            address_patterns = [
                r'[中国省市县区镇街道路巷弄号村乡大道园区工业区]+[\d\s]+[号号楼座层室]+',
                r'[中国省市县区镇街道路巷弄号村乡大道园区工业区]+[\d\s]+',
                r'[省市县区镇街道路巷弄号村乡大道园区工业区]+[\d\s]+'
            ]
            
            address_matches = []
            for pattern in address_patterns:
                matches = re.findall(pattern, query_text)
                if matches:
                    address_matches.extend(matches)
            
            logger.info(f"提取到地址匹配: {address_matches}")
            
            # 2. 提取地名关键词
            location_keywords = re.findall(r'[省市县区镇街道路巷弄号村乡大道园区工业区]+', query_text)
            logger.info(f"提取到地名关键词: {location_keywords}")
            
            # 执行混合检索获取候选结果
            hybrid_results = self._hybrid_search(
                query_text, 
                top_k=top_k * 5,  # 获取更多结果用于重排序
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            
            logger.info(f"获取到 {len(hybrid_results)} 个候选结果")
            
            if not hybrid_results:
                logger.warning("未获取到候选结果")
                return []
            
            # 地址匹配重排序
            address_reranked = []
            matched_docs = []
            
            for doc in hybrid_results:
                content = doc.get('content', '')
                doc_id = doc.get('id', '')
                original_score = doc.get('similarity', 0)
                
                logger.debug(f"处理文档: ID={doc_id}, 原始相似度={original_score}")
                
                # 计算地址匹配得分
                address_score = 0.0
                is_matched = False
                
                # 检查完整地址匹配
                if address_matches:
                    # 计算完整地址匹配得分
                    full_address_score = 0.0
                    for addr in address_matches:
                        if addr in content:
                            # 完整地址匹配，给予高分
                            full_address_score += 1.0 * (len(addr) / len(query_text))
                            is_matched = True
                            logger.info(f"文档 {doc_id} 完整地址匹配: {addr}")
                    
                    if full_address_score > 0:
                        address_score += full_address_score * 0.6
                
                # 检查地名关键词匹配
                if location_keywords:
                    # 计算地名关键词匹配得分
                    location_score = 0.0
                    matched_location_count = 0
                    for loc in location_keywords:
                        if loc in content:
                            matched_location_count += 1
                            logger.debug(f"文档 {doc_id} 匹配地名: {loc}")
                    
                    if matched_location_count > 0:
                        location_score = (matched_location_count / len(location_keywords)) * 0.4
                        address_score += location_score
                        if matched_location_count == len(location_keywords):
                            is_matched = True
                            logger.info(f"文档 {doc_id} 完全匹配所有地名关键词")
                
                # 额外检查：如果文档中包含多个地址关键词，增加得分
                content_location_keywords = re.findall(r'[省市县区镇街道路巷弄号村乡大道园区工业区]+', content)
                if len(content_location_keywords) >= 3:
                    address_score += 0.2
                    logger.debug(f"文档 {doc_id} 包含多个地址关键词，加分")
                
                # 结合原始得分和地址得分，地址得分权重更高
                final_score = (original_score * 0.5) + (address_score * 0.5)
                
                # 如果是精确匹配，给予更高的相似度
                if is_matched:
                    final_score = max(final_score, 0.9)  # 确保匹配的文档有较高的相似度
                    matched_docs.append(doc)
                
                doc_copy = doc.copy()
                doc_copy['similarity'] = final_score
                doc_copy['address_score'] = address_score
                doc_copy['original_score'] = original_score
                doc_copy['is_address_match'] = is_matched
                address_reranked.append(doc_copy)
            
            # 按最终得分排序
            address_reranked.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"匹配到 {len(matched_docs)} 个地址相关文档")
            
            # 应用最小相似度过滤
            if min_similarity is not None:
                filtered = [doc for doc in address_reranked if doc['similarity'] >= min_similarity]
                logger.info(f"应用最小相似度 {min_similarity} 后，剩余 {len(filtered)} 个文档")
                address_reranked = filtered
            
            # 调试：打印前10个结果
            if address_reranked:
                logger.info("前10个结果:")
                for i, doc in enumerate(address_reranked[:10]):
                    logger.info(f"  第{i+1}名: ID={doc.get('id')}, 相似度={doc.get('similarity'):.3f}, 匹配={doc.get('is_address_match')}")
            
            return address_reranked[:top_k]
            
        except Exception as e:
            logger.error(f"地址优化检索失败: {e}")
            # 失败时返回混合检索结果
            return self._hybrid_search(
                query_text, 
                top_k=top_k,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
    
    def _is_business_scope_query(self, query_text: str) -> bool:
        """判断是否为经营范围查询
        
        Args:
            _text: 
            
        Returns:
            : 是否为经营范围查询
"""
        # 经营范围关键词
        business_keywords = [
            '经营范围', '主要业务', '从事', '业务', '经营', '项目', '许可',
            '业务范围', '经营项目', '许可项目', '一般项目', '主营业务',
            '经营内容', '生产范围', '服务范围', '销售范围', '经营范围包括',
            '做', '从事', '生产', '销售', '提供', '研发', '业务为', '主要为',
            '及', '的公司', '的企业'
        ]
        
        # 检查是否包含业务范围相关关键词
        has_business_keyword = any(keyword in query_text for keyword in business_keywords)
        
        # 检查是否包含行业相关关键词 + 经营/从事等动词
        industry_verbs = ['经营', '从事', '生产', '销售', '提供', '研发', '做']
        has_industry_verb = any(verb in query_text for verb in industry_verbs)
        
        # 检查是否包含典型的经营范围查询模式
        has_business_pattern = bool(re.search(r'[经营从事生产销售提供研发做].*[业务项目范围]', query_text)) or \
                              bool(re.search(r'[业务项目范围].*[经营从事生产销售提供研发做]', query_text)) or \
                              bool(re.search(r'做.*的公司', query_text)) or \
                              bool(re.search(r'经营.*的企业', query_text))
        
        return has_business_keyword or has_industry_verb or has_business_pattern
    
    def _business_scope_enhanced_search(self, query_text: str, top_k: int, 
                                       where_filter: Optional[Dict] = None,
                                       min_similarity: Optional[float] = None) -> List[Dict]:
        """经营范围增强检索
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            
        Returns:
            List[Dict]: 检索结果
"""
        try:
            # 1. 提取多种关键词
            # 提取主要业务关键词
            business_keywords = jieba.analyse.extract_tags(query_text, topK=20, withWeight=True)
            # 提取名词性关键词
            noun_keywords = jieba.analyse.extract_tags(query_text, topK=15, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nt', 'nw'))
            # 提取动词性关键词
            verb_keywords = jieba.analyse.extract_tags(query_text, topK=10, withWeight=True, allowPOS=('v', 'vd', 'vn'))
            # 提取TextRank关键词作为补充
            textrank_keywords = jieba.analyse.textrank(query_text, topK=15, withWeight=True)
            # 合并关键词，去重
            all_keywords = {}
            for word, weight in business_keywords + noun_keywords + verb_keywords + textrank_keywords:
                if word in all_keywords:
                    all_keywords[word] += weight
                else:
                    all_keywords[word] = weight
            # 转换回列表格式
            combined_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:30]
            
            # 2. 执行多种检索获取候选结果
            # 混合检索
            hybrid_results = self._hybrid_search(
                query_text, 
                top_k=top_k * 4,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            
            # 向量检索作为补充
            vector_results = self._vector_search_optimized(
                query_text, 
                top_k=top_k * 4,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            
            # 合并结果，去重
            all_results = []
            seen_ids = set()
            
            for doc in hybrid_results + vector_results:
                if doc.get('id') not in seen_ids:
                    seen_ids.add(doc.get('id'))
                    all_results.append(doc)
            
            if not all_results:
                return []
            
            # 3. 业务范围重排序
            business_reranked = []
            matched_docs = []
            
            logger.info(f"开始经营范围重排序，候选文档数: {len(all_results)}")
            logger.info(f"提取关键词数: {len(combined_keywords)}")
            logger.info(f"前10个关键词: {[(word, round(weight, 2)) for word, weight in combined_keywords[:10]]}")
            
            for doc in all_results:
                content = doc.get('content', '')
                original_score = doc.get('similarity', 0)
                
                # 计算业务关键词匹配得分
                business_score = 0.0
                is_matched = False
                
                # 3.1 综合关键词匹配
                if combined_keywords:
                    matched_combined = []
                    for word, weight in combined_keywords:
                        if word in content:
                            matched_combined.append(weight)
                    
                    if matched_combined:
                        # 基础关键词得分
                        keyword_base_score = sum(matched_combined) / len(matched_combined) * 0.5
                        # 关键词覆盖率增强
                        coverage = len(matched_combined) / len(combined_keywords) if combined_keywords else 0
                        coverage_bonus = coverage * 0.3
                        # 关键词权重分布增强
                        top_5_matched = [w for w in matched_combined[:5]]
                        top_5_bonus = sum(top_5_matched) / 5 * 0.2 if top_5_matched else 0
                        
                        combined_keyword_score = keyword_base_score + coverage_bonus + top_5_bonus
                        business_score += combined_keyword_score
                        
                        # 检查是否匹配到足够多的关键词
                        if coverage >= 0.3 or len(matched_combined) >= 5:
                            is_matched = True
                            matched_docs.append(doc)
                
                # 3.2 特殊业务范围标记增强
                # 检查是否包含"经营范围"、"经营项目"等标记词
                business_markers = ['经营范围', '经营项目', '许可项目', '一般项目', '主营业务']
                marker_matched = False
                for marker in business_markers:
                    if marker in content:
                        business_score += 0.2
                        marker_matched = True
                        break
                
                # 3.3 业务动词匹配增强
                # 检查是否包含经营相关动词
                action_verbs = ['经营', '从事', '生产', '销售', '提供', '研发', '制造', '服务']
                verb_matched_count = sum(1 for verb in action_verbs if verb in content)
                if verb_matched_count > 0:
                    business_score += verb_matched_count * 0.1
                
                # 3.4 行业术语匹配增强
                # 检查是否包含行业相关术语
                industry_terms = []
                for word, _ in combined_keywords:
                    if len(word) > 2:  # 长词更可能是行业术语
                        industry_terms.append(word)
                
                if industry_terms:
                    industry_matched = sum(1 for term in industry_terms if term in content)
                    industry_score = (industry_matched / len(industry_terms)) * 0.2
                    business_score += industry_score
                
                # 4. 结合原始得分和业务得分
                final_score = (original_score * 0.4) + (business_score * 0.6)
                
                # 如果是匹配到的文档，给予更高的相似度
                if is_matched or marker_matched:
                    final_score = max(final_score, 0.85)  # 确保匹配的文档有较高的相似度
                
                doc_copy = doc.copy()
                doc_copy['similarity'] = final_score
                doc_copy['business_score'] = business_score
                doc_copy['original_score'] = original_score
                doc_copy['is_business_match'] = is_matched
                doc_copy['marker_matched'] = marker_matched
                doc_copy['keyword_coverage'] = len(matched_combined) / len(combined_keywords) if combined_keywords and 'matched_combined' in locals() else 0
                business_reranked.append(doc_copy)
            
            # 5. 按最终得分排序
            business_reranked.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"重排序后文档数: {len(business_reranked)}")
            logger.info(f"匹配到的相关文档数: {len(matched_docs)}")
            
            # 6. 应用最小相似度过滤
            if min_similarity is not None:
                filtered = [doc for doc in business_reranked if doc['similarity'] >= min_similarity]
                logger.info(f"应用最小相似度 {min_similarity} 后，剩余 {len(filtered)} 个文档")
                business_reranked = filtered
            
            # 7. 调试：打印前10个结果
            if business_reranked:
                logger.info("前10个经营范围查询结果:")
                for i, doc in enumerate(business_reranked[:10]):
                    logger.info(f"  第{i+1}名: ID={doc.get('id')}, 相似度={doc.get('similarity'):.3f}, 匹配={doc.get('is_business_match')}, 覆盖率={doc.get('keyword_coverage'):.2f}")
            
            return business_reranked[:top_k]
            
        except Exception as e:
            logger.error(f"经营范围增强检索失败: {e}")
            # 失败时返回混合检索结果
            return self._hybrid_search(
                query_text, 
                top_k=top_k,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
    
    def _rerank_results(self, query_text: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """结果重排序：基于文档内容和元数据进行二次排序
"""
        if not docs:
            return []
        
        try:
            # 提取查询关键词
            query_keywords = set(jieba.cut(query_text))
            
            # 为每个文档计算重排序得分
            reranked_docs = []
            for doc in docs:
                # 原始相似度
                original_score = doc.get('similarity', 0)
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # 重排序得分因子
                content_score = 0.0
                metadata_score = 0.0
                length_score = 0.0
                
                # 1. 内容得分：基于关键词在文档中的位置
                if content:
                    # 关键词出现在文档开头的权重更高
                    content_lower = content.lower()
                    query_lower = query_text.lower()
                    
                    # 计算查询文本在文档中的匹配位置
                    if query_lower in content_lower:
                        pos = content_lower.index(query_lower)
                        # 位置越靠前，得分越高
                        position_weight = max(0, 1
"""测试连接和检索功能
"""test_queries = [
            "人工智能",
            "机器学习",
            "深度学习"
        ]
        
        results = []
        for query in test_queries:
            try:
                result = self.query(query, top_k=3)
                results.append({
                    "query": query,
                    "time": result.retrieval_time,
                    "documents": result.total_retrieved,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        # 获取集合统计
        try:
            count = self.collection.count()
            collection_stats = {
                "document_count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            collection_stats = {"error": str(e)}
        
        return {
            "test_results": results,
            "collection_stats": collection_stats,
            "model_loaded": self.embedding_model is not None,
            "db_connected": self.chroma_client is not None
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息
"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "db_path": self.db_path,
                "error": str(e)
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
"""
        if hasattr(self, 'query_cache'):
            return self.query_cache.get_stats()
        return {
            "error": "Cache not initialized"
        }
    
    def clear_cache(self) -> None:
        """清空查询缓存
"""
        if hasattr(self, 'query_cache'):
            self.query_cache.clear()
            logger.info("查询缓存已清空")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标
"""
        cache_stats = self.get_cache_stats()
        
        return {
            "cache": cache_stats,
            "collection": self.get_collection_stats()
        }
    
    def close(self):
        """关闭资源
"""
        logger.info("查询器已关闭")

# 测试代码
if __name__ == "__main__":
    # 初始化查询器
    query_tool = OptimizedVectorDBQuery()
    
    # 测试连接
    print("=" * 60)
    print("测试连接和检索功能")
    print("=" * 60)
    
    test_result = query_tool.test_connection()
    
    print("\n连接测试结果:")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    
    if test_result["model_loaded"] and test_result["db_connected"]:
        # 执行示例查询
        test_query = "向量数据库"
        print(f"\n执行示例查询: '{test_query}'")
        
        start_time = time.time()
        result = query_tool.query(test_query, top_k=5)
        elapsed = time.time()
"""向量数据库查询器
"""import os
import json
import time
import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass, field
from datetime import datetime
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
        """转换为字典格式
"""
        return {
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "retrieval_time": self.retrieval_time,
            "avg_similarity": self.avg_similarity,
            "query_id": self.query_id,
            "retrieval_method": self.retrieval_method,
            "reranked": self.reranked,
            "retrieved_documents": self.retrieved_documents
        }

class QueryCache:
    """查询缓存系统
"""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def _generate_key(self, query: str, params: Dict) -> str:
        """生成缓存键
"""
        key_str = f"{query}_{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, params: Dict) -> Optional[Any]:
        """获取缓存结果
"""
        key = self._generate_key(query, params)
        return self.cache.get(key)
    
    def set(self, query: str, params: Dict, result: Any) -> None:
        """设置缓存结果
"""
        key = self._generate_key(query, params)
        
        if key in self.cache:
            self.cache.pop(key)
        
        self.cache[key] = result
        
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)

class OptimizedVectorDBQuery:
    """向量数据库查询器
"""
    
    def __init__(self, 
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = None):
        
        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        
        # 自动选择设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("初始化版向量数据库查询器")
        logger.info(f"设备: {self.device}")
        
        # 初始化组件
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_cache = QueryCache(max_size=200)
        
        # 配置
        self.config = {
            "default_top_k": 10,
            "enable_cache": True,
            "enable_hybrid_search": True,  # 启用混合检索
        }
        
        # 初始化所有组件
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化所有组件
"""
        try:
            # 1. 初始化embedding模型
            logger.info("加载Embedding模型...")
            self.embedding_model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True
            )
            
            # 2. 初始化ChromaDB
            logger.info("连接ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取集合（如果不存在会创建）
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except:
                # 如果集合不存在，创建新集合
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            logger.info("✅ 初始化完成")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def query(self, 
              query_text: str, 
              top_k: int = None,
              where_filter: Optional[Dict[str, Any]] = None,
              min_similarity: Optional[float] = 0.0,
              return_format: str = "structured") -> Union[QueryResult, List[Dict[str, Any]]]:
        """查询
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            return_format:
"""
        start_time = time.time()
        
        # 使用默认值
        if top_k is None:
            top_k = self.config["default_top_k"]
        
        # 检查缓存
        cache_params = {
            "query": query_text,
            "top_k": top_k,
            "where_filter": where_filter,
            "min_similarity": min_similarity
        }
        
        if self.config["enable_cache"]:
            cached_result = self.query_cache.get(query_text, cache_params)
            if cached_result:
                if return_format == "list":
                    return cached_result["retrieved_documents"]
                else:
                    return QueryResult(**cached_result)
        
        try:
            # 执行向量检索
            retrieved_docs = self._vector_search_optimized(
                query_text, 
                top_k=top_k,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            
            # 取top_k个结果
            final_docs = retrieved_docs[:top_k]
            
            # 计算统计信息
            retrieval_time = time.time()
"""向量检索（兼容新版ChromaDB）
"""try:
            # 向量化
            query_embedding = self.embedding_model.encode(
                query_text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # 执行查询
"""备选向量检索方法
"""try:
            # 向量化
            query_embedding = self.embedding_model.encode(
                query_text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # 尝试不同的API调用方式
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            # 尝试不同的include参数组合
            include_options = [
                ["documents", "metadatas", "distances"],
                ["documents", "distances"],
                ["documents", "metadatas"],
                ["documents"]
            ]
            
            for include_option in include_options:
                try:
                    query_params["include"] = include_option
                    results = self.collection.query(**query_params)
                    
                    # 处理结果
                    retrieved_docs = []
                    if results['documents'] and results['documents'][0]:
                        for i, doc_content in enumerate(results['documents'][0]):
                            # 获取相似度
                            if 'distances' in results and results['distances'][0]:
                                distance = results['distances'][0][i]
                                similarity = 1
"""测试连接和检索功能
"""
        test_queries = [
            "人工智能",
            "机器学习",
            "深度学习"
        ]
        
        results = []
        for query in test_queries:
            try:
                result = self.query(query, top_k=3)
                results.append({
                    "query": query,
                    "time": result.retrieval_time,
                    "documents": result.total_retrieved,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        # 获取集合统计
        try:
            count = self.collection.count()
            collection_stats = {
                "document_count": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            collection_stats = {"error": str(e)}
        
        return {
            "test_results": results,
            "collection_stats": collection_stats,
            "model_loaded": self.embedding_model is not None,
            "db_connected": self.chroma_client is not None
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息
"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "db_path": self.db_path,
                "error": str(e)
            }
    
    def close(self):
        """关闭资源
"""
        logger.info("查询器已关闭")

# 测试代码
if __name__ == "__main__":
    # 初始化查询器
    query_tool = OptimizedVectorDBQuery()
    
    # 测试连接
    print("=" * 60)
    print("测试连接和检索功能")
    print("=" * 60)
    
    test_result = query_tool.test_connection()
    
    print("\n连接测试结果:")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    
    if test_result["model_loaded"] and test_result["db_connected"]:
        # 执行示例查询
        test_query = "向量数据库"
        print(f"\n执行示例查询: '{test_query}'")
        
        start_time = time.time()
        result = query_tool.query(test_query, top_k=5)
        elapsed = time.time() - start_time
        
        print(f"查询耗时: {elapsed:.3f}s")
        print(f"返回文档数: {result.total_retrieved}")
        print(f"平均相似度: {result.avg_similarity:.3f}")
        
        if result.retrieved_documents:
            print("\n检索到的文档:")
            for i, doc in enumerate(result.retrieved_documents):
                print(f"  文档 {i+1}: {doc['content_preview']}")
                print(f"    相似度: {doc['similarity']:.3f}")
                print(f"    元数据: {doc['metadata']}")
                print()
    
    # 关闭查询器
    query_tool.close()