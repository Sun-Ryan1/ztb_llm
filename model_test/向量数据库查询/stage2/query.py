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
            "default_top_k": 10,  # 增加默认返回结果数量
            "enable_cache": True,
            "enable_hybrid_search": True,  # 启用混合检索（需要实现）
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