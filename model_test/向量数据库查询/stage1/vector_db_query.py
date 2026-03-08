#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询器
"""

import os
import json
import time
import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import OrderedDict
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s
"""查询结果数据结构
"""转换为字典格式"""result = {
            "query": self.query,
            "total_retrieved": self.total_retrieved,
            "retrieval_time": self.retrieval_time,
            "embedding_time": self.embedding_time,
            "query_time": self.query_time,
            "avg_similarity": self.avg_similarity,
            "query_id": self.query_id or f"query_{int(time.time())}",
            "retrieved_documents": self.retrieved_documents
        }
        if self.intent_type:
            result["intent_type"] = self.intent_type
        if self.optimized_query:
            result["optimized_query"] = self.optimized_query
        return result
    
    def to_json(self, filepath: str = None) -> str:
"""转换为JSON格式"""json_str = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str
    
    def to_markdown(self) -> str:
"""转换为Markdown格式"""md = f"# 查询结果\n\n"
        md += f"**查询**: {self.query}\n\n"
        md += f"**耗时**: {self.retrieval_time:.3f}s | "
        md += f"**返回文档数**: {self.total_retrieved} | "
        md += f"**平均相似度**: {self.avg_similarity:.3f}\n\n"
        
        if self.intent_type:
            md += f"**查询意图**: {self.intent_type}\n\n"
        
        if self.optimized_query and self.optimized_query != self.query:
            md += f"**优化后查询**: {self.optimized_query}\n\n"
        
        if self.retrieved_documents:
            md += "## 检索结果\n\n"
            md += "| 排名 | 相似度 | 内容预览 | 类型 |\n"
            md += "|------|--------|----------|------|\n"
            
            for doc in self.retrieved_documents[:10]:  # 只显示前10个
                doc_type = doc['metadata'].get('type', '未知')
                preview = doc['content_preview'].replace('\n', ' ').replace('|', '\\|')
                md += f"| {doc['rank']} | {doc['similarity']:.3f} | {preview[:80]}... | {doc_type} |\n"
        
        return md
    
    def to_html(self) -> str:
"""转换为HTML格式"""html = f
"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>查询结果
"""if self.intent_type:
            html += f'<p><strong>查询意图:</strong> {self.intent_type}</p>'
        
        html += "</div>"
        
        if self.retrieved_documents:
            html += "<h3>检索结果</h3>"
            for doc in self.retrieved_documents:
                # 根据相似度设置CSS类
                similarity_class = "similarity-low"
                if doc['similarity'] > 0.7:
                    similarity_class = "similarity-high"
                elif doc['similarity'] > 0.4:
                    similarity_class = "similarity-medium"
                
                html += f'<div class="result {similarity_class}">'
                html += f'<p><strong>#{doc["rank"]}</strong> 相似度: {doc["similarity"]:.3f}</p>'
                html += f'<p>{doc["content_preview"]}</p>'
                
                # 显示元数据
                if doc['metadata']:
                    html += '<div class="metadata">'
                    for key, value in list(doc['metadata'].items())[:3]:
                        html += f'<span>{key}: {value} &nbsp;&nbsp;</span>'
                    html += '</div>'
                
                html += '</div>'
        
        html +=
"""
        </body>
        </html>
        """return html
    
    def __str__(self) -> str:
"""字符串表示"""return f"Query: '{self.query[:50]}...' | Docs: {self.total_retrieved} | Time: {self.retrieval_time:.3f}s | Avg Sim: {self.avg_similarity:.3f}"

class QueryIntentRecognizer:
"""查询意图识别器"""INTENT_PATTERNS = [
        # 公司查询
        (r'(.*?)(公司|有限公司|有限责任公司|股份有限公司)$', 'company_exact'),
        (r'(.*)(的)?(法定代表人|法人代表)(是|为)?', 'company_legal_representative'),
        (r'(.*)(的)?(统一社会信用代码|信用代码)(是|多少)?', 'company_credit_code'),
        (r'(.*)(的)?(注册地址|地址)(是|在哪里)?', 'company_address'),
        (r'(.*)(的)?(经营范围|业务范围)(是|包括)?', 'company_business_scope'),
        (r'(.*)(的)?(注册资本|注册资金)(是|多少)?', 'company_capital'),
        (r'(.*)(的)?(公司类型|企业类型)(是|什么)?', 'company_type'),
        
        # 地点查询
        (r'(.*)(在|位于)(哪里|何处)', 'location_query'),
        
        # 通用查询
        (r'什么是(.+)', 'definition_query'),
        (r'(.+)是什么', 'definition_query'),
        (r'如何(.+)', 'howto_query'),
    ]
    
    @staticmethod
    def recognize(query: str) -> Dict[str, Any]:
"""识别查询意图"""query = query.strip()
        
        for pattern, intent_type in QueryIntentRecognizer.INTENT_PATTERNS:
            match = re.match(pattern, query)
            if match:
                return {
                    'type': intent_type,
                    'original_query': query,
                    'confidence': 0.9,
                    'matched_pattern': pattern
                }
        
        # 默认意图
        return {
            'type': 'general_query',
            'original_query': query,
            'confidence': 0.5,
            'matched_pattern': None
        }
    
    @staticmethod
    def optimize_query(query: str, intent: Dict[str, Any]) -> str:
"""根据意图优化查询文本"""query = query.strip()
        
        # 移除常见查询词
        if intent['type'].startswith('company_'):
            # 移除"的"和疑问词
            query = re.sub(r'(的|是|什么|多少|在哪里|为)', '', query)
        
        # 公司名称标准化
        if intent['type'] == 'company_exact':
            # 移除空格，标准化公司名称
            query = re.sub(r'\s+', '', query)
        
        return query.strip()

class QueryOptimizer:
"""查询优化器"""
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """预处理查询文本
"""
        # 1. 去除多余空格
        query = re.sub(r'\s+', ' ', query).strip()
        
        # 2. 提取中文关键词（移除标点符号）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', query)
        if chinese_chars:
            query = ' '.join(chinese_chars)
        
        # 3. 公司名称特殊处理
        if '公司' in query:
            # 移除常见的行政区划前缀（可选）
            query = re.sub(r'^(.*?[省市县区])+', '', query)
        
        return query
    
    @staticmethod
    def expand_query(query: str, intent_type: str = None) -> List[str]:
        """查询扩展，生成多个查询变体
"""
        variants = [query]
        
        # 根据意图类型扩展
        if intent_type and intent_type.startswith('company_'):
            # 公司查询扩展
            if '公司' in query:
                # 变体1：移除"有限公司"等后缀
                variant = re.sub(r'(有限|股份)?公司$', '', query)
                if variant != query:
                    variants.append(variant.strip())
                
                # 变体2：添加"的"后缀
                variants.append(query + "的")
        
        # 基于同义词扩展（简化版）
        synonyms = {
            '法定代表人': ['法人代表', '法人'],
            '注册地址': ['地址', '所在地'],
            '统一社会信用代码': ['信用代码', '社会信用代码'],
        }
        
        for key, syn_list in synonyms.items():
            if key in query:
                for syn in syn_list:
                    variant = query.replace(key, syn)
                    variants.append(variant)
        
        return list(set(variants))  # 去重

class SmartResultProcessor:
    """智能结果处理器
"""
    
    @staticmethod
    def group_by_company(documents: List[Dict]) -> Dict[str, List[Dict]]:
        """按公司分组文档
"""
        groups = {}
        
        for doc in documents:
            company_name = doc['metadata'].get('company_name')
            if not company_name:
                # 尝试从内容中提取公司名称
                content = doc['content']
                company_match = re.search(r'([\u4e00-\u9fff\s]+(?:公司|有限公司))', content)
                if company_match:
                    company_name = company_match.group(1).strip()
                else:
                    company_name = '未知公司'
            
            # 标准化公司名称（移除空格）
            company_name = re.sub(r'\s+', '', company_name)
            
            if company_name not in groups:
                groups[company_name] = []
            
            groups[company_name].append(doc)
        
        return groups
    
    @staticmethod
    def deduplicate_documents(documents: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """基于内容哈希去重文档
"""
        seen_hashes = set()
        deduplicated = []
        
        for doc in documents:
            # 生成内容哈希
            content = doc['content']
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(doc)
            else:
                # 保留相似度更高的版本
                for i, existing_doc in enumerate(deduplicated):
                    existing_hash = hashlib.md5(existing_doc['content'].encode('utf-8')).hexdigest()[:16]
                    if existing_hash == content_hash and doc['similarity'] > existing_doc['similarity']:
                        deduplicated[i] = doc
        
        return deduplicated
    
    @staticmethod
    def filter_by_similarity(documents: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """基于相似度阈值过滤文档
"""
        if not documents:
            return []
        
        # 自动阈值检测
        if threshold is None:
            similarities = [doc['similarity'] for doc in documents]
            if len(similarities) >= 3:
                # 计算相似度的标准差，设置自适应阈值
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                threshold = max(mean_sim
"""LRU缓存实现
"""def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str):
        """获取缓存值
"""
        if key not in self.cache:
            return None
        
        # 移动到最后（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """设置缓存值
"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # 如果超过容量，移除最久未使用的
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def clear(self):
        """清空缓存
"""
        self.cache.clear()
    
    def size(self) -> int:
        """获取缓存大小
"""
        return len(self.cache)

def monitor_performance(func):
    """性能监控装饰器
"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.time()
"""增强版向量数据库查询器
"""def __init__(self, 
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = None,
                 config: Dict = None):
        """初始化增强版向量数据库查询器
        
        Args:
            _path: 
            model_path: 
            collection_name: 
            device: ，自动检测GPU
            config:
"""
        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 合并配置
        self.config = {
            "default_top_k": 5,
            "min_similarity_threshold": 0.0,
            "max_top_k": 100,
            "enable_cache": True,
            "cache_size": 1000,
            "enable_intent_recognition": True,
            "enable_query_optimization": True,
            "auto_filter_threshold": True,
            "enable_grouping": True,
            "enable_deduplication": True,
            "thread_pool_size": 4
        }
        
        if config:
            self.config.update(config)
        
        logger.info("=" * 60)
        logger.info("初始化增强版向量数据库查询器")
        logger.info("=" * 60)
        logger.info(f"数据库路径: {self.db_path}")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"集合名称: {self.collection_name}")
        logger.info(f"运行设备: {self.device}")
        logger.info(f"配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
        # 初始化组件
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
        # 查询统计
        self.query_stats = {
            "total_queries": 0,
            "total_retrieved": 0,
            "total_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 查询历史
        self.query_history = []
        
        # 初始化缓存（使用LRU缓存）
        self.query_cache = LRUCache(capacity=self.config["cache_size"])
        
        # 线程池用于异步操作
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config["thread_pool_size"])
        
        # 性能指标
        self.performance_metrics = []
        
        # 初始化所有组件
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化所有组件
"""
        try:
            # 1. 初始化embedding模型
            self._initialize_embedding_model()
            
            # 2. 初始化ChromaDB
            self._initialize_chromadb()
            
            # 3. 验证连接状态
            self._validate_connections()
            
            logger.info("✅ 所有组件初始化成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise
    
    def _initialize_embedding_model(self) -> None:
        """初始化embedding模型
"""
        logger.info("正在加载embedding模型...")
        
        try:
            # 检查模型路径
            if not os.path.exists(self.model_path):
                logger.warning(f"模型路径不存在: {self.model_path}")
                logger.info("尝试从ModelScope下载模型...")
                # 尝试从ModelScope下载
                try:
                    from modelscope import snapshot_download
                    self.model_path = snapshot_download("BAAI/bge-m3", 
                                                      cache_dir=os.path.dirname(self.model_path))
                    logger.info(f"模型下载完成: {self.model_path}")
                except ImportError:
                    logger.error("modelscope未安装，请安装: pip install modelscope")
                    raise
                except Exception as e:
                    logger.error(f"下载模型失败: {e}")
                    raise
            
            # 加载模型
            self.embedding_model = SentenceTransformer(
                self.model_path,
                device=self.device
            )
            
            # 测试模型
            test_embeddings = self.embedding_model.encode(["测试文本"])
            embedding_dim = test_embeddings.shape[1]
            
            logger.info(f"✅ Embedding模型加载成功")
            logger.info(f"   向量维度: {embedding_dim}")
            logger.info(f"   模型: {os.path.basename(self.model_path)}")
            
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            
            # 尝试备用方案
            try:
                logger.info("尝试加载默认模型: BAAI/bge-m3")
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-m3",
                    device=self.device
                )
                logger.info("✅ 备用模型加载成功")
            except Exception as e2:
                logger.error(f"备用模型也失败: {e2}")
                raise
    
    def _initialize_chromadb(self) -> None:
        """初始化ChromaDB连接
"""
        logger.info("正在连接ChromaDB...")
        
        try:
            # 检查数据库路径
            if not os.path.exists(self.db_path):
                logger.error(f"数据库路径不存在: {self.db_path}")
                raise FileNotFoundError(f"数据库路径不存在: {self.db_path}")
            
            # 创建ChromaDB客户端
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # 获取集合
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"✅ ChromaDB连接成功")
                logger.info(f"   集合: {self.collection_name}")
            except Exception as e:
                logger.error(f"获取集合失败: {e}")
                logger.info("可用集合列表:")
                collections = self.chroma_client.list_collections()
                for col in collections:
                    logger.info(f"
"""验证所有连接状态
"""logger.info("验证组件连接状态...")
        
        checks = {
            "Embedding模型": self.embedding_model is not None,
            "ChromaDB客户端": self.chroma_client is not None,
            "集合": self.collection is not None
        }
        
        all_ok = True
        for component, status in checks.items():
            if status:
                logger.info(f"✅ {component}: 正常")
            else:
                logger.error(f"❌ {component}: 异常")
                all_ok = False
        
        if not all_ok:
            raise ConnectionError("组件连接验证失败")
        
        # 测试查询验证
        try:
            test_count = self.collection.count()
            logger.info(f"✅ 集合文档数量: {test_count}")
        except Exception as e:
            logger.error(f"❌ 集合查询测试失败: {e}")
            raise
        
        logger.info("✅ 所有组件验证通过")
    
    def _generate_query_id(self, query: str) -> str:
        """生成查询ID
"""
        timestamp = int(time.time())
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
        return f"q_{timestamp}_{query_hash}"
    
    def _embed_text(self, text: str) -> np.ndarray:
        """将文本转换为向量
"""
        try:
            start_time = time.time()
            embedding = self.embedding_model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embedding_time = time.time()
"""执行增强版向量查询
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            return_format: 
            enable_smart_processing: 
            
        Returns:
            查询结果
"""if top_k is None:
            top_k = self.config["default_top_k"]
        
        # 更新查询统计
        self.query_stats["total_queries"] += 1
        
        # 生成查询ID
        query_id = self._generate_query_id(query_text)
        
        # 缓存键
        cache_key = f"{query_text}_{top_k}_{str(where_filter)}_{min_similarity}_{enable_smart_processing}"
        
        # 检查缓存
        if self.config["enable_cache"]:
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                logger.debug(f"使用缓存查询: {query_text[:50]}...")
                self.query_stats["cache_hits"] += 1
                
                # 记录查询历史
                self._record_query_history(query_text, cached_result["retrieval_time"], 
                                          cached_result["total_retrieved"], True)
                
                # 根据格式返回
                if return_format == "list":
                    return cached_result["retrieved_documents"]
                else:
                    return QueryResult(**cached_result)
        
        self.query_stats["cache_misses"] += 1
        
        start_time = time.time()
        
        try:
            logger.info(f"[{query_id}] 处理查询: '{query_text}'")
            
            # 智能处理流程
            optimized_query = query_text
            intent_type = None
            
            if enable_smart_processing:
                # 1. 查询意图识别
                if self.config["enable_intent_recognition"]:
                    intent = QueryIntentRecognizer.recognize(query_text)
                    intent_type = intent['type']
                    
                    # 2. 查询优化
                    if self.config["enable_query_optimization"]:
                        optimized_query = QueryIntentRecognizer.optimize_query(query_text, intent)
                        optimized_query = QueryOptimizer.preprocess_query(optimized_query)
                        logger.debug(f"优化后查询: '{optimized_query}'")
            
            # 3. 文本向量化
            embedding_start = time.time()
            query_embedding = self._embed_text(optimized_query)
            embedding_time = time.time()
"""智能查询：返回结构化信息
        
        Args:
            _text: 
            top_k: 
            where_filter: 
            min_similarity: 
            
        Returns:
            结构化查询结果
"""
        # 执行查询
        result = self.query(
            query_text, 
            top_k=top_k,
            where_filter=where_filter,
            min_similarity=min_similarity,
            enable_smart_processing=True
        )
        
        # 按公司分组
        grouped_docs = {}
        if result.retrieved_documents:
            grouped_docs = SmartResultProcessor.group_by_company(result.retrieved_documents)
        
        # 生成公司摘要
        company_summaries = []
        for company_name, docs in grouped_docs.items():
            # 计算公司平均相似度
            company_similarities = [doc['similarity'] for doc in docs]
            avg_similarity = np.mean(company_similarities) if company_similarities else 0
            
            # 提取关键信息
            summary = {
                'company_name': company_name,
                'document_count': len(docs),
                'avg_similarity': avg_similarity,
                'max_similarity': max(company_similarities) if company_similarities else 0,
                'info_types': list(set([doc['metadata'].get('type', '未知') for doc in docs])),
                'key_info': {}
            }
            
            # 从文档中提取关键信息
            for doc in docs:
                metadata = doc['metadata']
                if 'legal_representative' in metadata:
                    summary['key_info']['法定代表人'] = metadata['legal_representative']
                if 'credit_code' in metadata:
                    summary['key_info']['统一社会信用代码'] = metadata['credit_code']
                if 'registration_address' in metadata:
                    summary['key_info']['注册地址'] = metadata['registration_address']
                if 'company_type' in metadata:
                    summary['key_info']['公司类型'] = metadata['company_type']
            
            company_summaries.append(summary)
        
        # 按最大相似度排序
        company_summaries.sort(key=lambda x: x['max_similarity'], reverse=True)
        
        return {
            'query': result.query,
            'intent_type': result.intent_type,
            'optimized_query': result.optimized_query,
            'performance': {
                'retrieval_time': result.retrieval_time,
                'embedding_time': result.embedding_time,
                'query_time': result.query_time,
                'avg_similarity': result.avg_similarity
            },
            'statistics': {
                'total_documents': result.total_retrieved,
                'unique_companies': len(company_summaries),
                'companies_found': len(grouped_docs)
            },
            'companies': company_summaries,
            'grouped_documents': grouped_docs,
            'raw_result': result.to_dict() if isinstance(result, QueryResult) else result
        }
    
    def _record_query_history(self, query_text: str, duration: float, 
                             retrieved_count: int, from_cache: bool) -> None:
        """记录查询历史
"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query_text,
            "duration": duration,
            "retrieved_count": retrieved_count,
            "from_cache": from_cache
        }
        self.query_history.append(history_entry)
        
        # 限制历史记录大小
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def _cache_query_result(self, cache_key: str, query_result: QueryResult) -> None:
        """缓存查询结果
"""
        # 检查缓存大小
        if self.query_cache.size() >= self.config["cache_size"]:
            # LRU缓存会自动处理
            pass
        
        # 转换为可缓存的字典格式
        cache_data = {
            "query": query_result.query,
            "retrieved_documents": query_result.retrieved_documents,
            "retrieval_time": query_result.retrieval_time,
            "total_retrieved": query_result.total_retrieved,
            "embedding_time": query_result.embedding_time,
            "query_time": query_result.query_time,
            "avg_similarity": query_result.avg_similarity,
            "query_id": query_result.query_id,
            "intent_type": query_result.intent_type,
            "optimized_query": query_result.optimized_query,
            "cached_at": time.time()
        }
        
        # 添加到缓存
        self.query_cache.put(cache_key, cache_data)
    
    def batch_query(self, 
                   query_texts: List[str], 
                   top_k: int = 5,
                   where_filter: Optional[Dict[str, Any]] = None,
                   min_similarity: Optional[float] = None,
                   parallel: bool = True) -> List[QueryResult]:
        """批量查询
        
        Args:
            _texts: 
            top_k: 
            where_filter: 
            min_similarity: 
            parallel: 
            
        Returns:
"""
        logger.info(f"开始批量查询，共 {len(query_texts)} 个查询")
        
        if parallel and len(query_texts) > 1:
            # 并行执行
            with ThreadPoolExecutor(max_workers=self.config["thread_pool_size"]) as executor:
                futures = []
                for query_text in query_texts:
                    future = executor.submit(
                        self.query,
                        query_text,
                        top_k,
                        where_filter,
                        min_similarity
                    )
                    futures.append(future)
                
                results = [future.result() for future in futures]
        else:
            # 串行执行
            results = []
            for query_text in query_texts:
                result = self.query(query_text, top_k, where_filter, min_similarity)
                results.append(result)
        
        successful = len([r for r in results if r.total_retrieved > 0])
        logger.info(f"批量查询完成，成功 {successful}/{len(results)}")
        return results
    
    async def async_batch_query(self, 
                               query_texts: List[str], 
                               top_k: int = 5,
                               where_filter: Optional[Dict[str, Any]] = None,
                               min_similarity: Optional[float] = None) -> List[QueryResult]:
        """异步批量查询
"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行批量查询
        results = await loop.run_in_executor(
            self.thread_pool,
            lambda: self.batch_query(
                query_texts, 
                top_k, 
                where_filter, 
                min_similarity, 
                parallel=True
            )
        )
        
        return results
    
    def query_with_multiple_filters(self, 
                                   query_text: str,
                                   top_k: int = 5,
                                   where_filters: List[Dict[str, Any]] = None,
                                   min_similarity: Optional[float] = None) -> Dict[str, QueryResult]:
        """使用多个过滤条件进行查询
        
        Args:
            _text: 
            top_k: 
            where_filters: 
            min_similarity: 
            
        Returns:
            每个过滤条件的查询结果
"""
        if where_filters is None:
            where_filters = []
        
        results = {}
        for i, filter_condition in enumerate(where_filters):
            filter_name = f"filter_{i}" if "name" not in filter_condition else filter_condition["name"]
            logger.info(f"执行查询: '{query_text}'，过滤条件: {filter_condition}")
            
            result = self.query(query_text, top_k, filter_condition, min_similarity)
            results[filter_name] = result
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息
"""
        try:
            count = self.collection.count()
            
            # 获取所有文档的元数据
            all_docs = self.collection.get(limit=min(1000, count))
            
            # 分析元数据分布
            metadata_stats = {}
            if all_docs["metadatas"]:
                for metadata in all_docs["metadatas"]:
                    for key, value in metadata.items():
                        if key not in metadata_stats:
                            metadata_stats[key] = {}
                        
                        if value not in metadata_stats[key]:
                            metadata_stats[key][value] = 0
                        metadata_stats[key][value] += 1
            
            return {
                "total_documents": count,
                "metadata_fields": list(metadata_stats.keys()),
                "metadata_distribution": metadata_stats,
                "sample_size": min(1000, count)
            }
            
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {}
    
    def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息
"""
        avg_time = (self.query_stats["total_time"] / 
                   self.query_stats["successful_queries"]) if self.query_stats["successful_queries"] > 0 else 0
        
        cache_hit_rate = (self.query_stats["cache_hits"] / 
                         (self.query_stats["cache_hits"] + self.query_stats["cache_misses"]) * 100 
                         if (self.query_stats["cache_hits"] + self.query_stats["cache_misses"]) > 0 else 0)
        
        return {
            "total_queries": self.query_stats["total_queries"],
            "successful_queries": self.query_stats["successful_queries"],
            "failed_queries": self.query_stats["failed_queries"],
            "success_rate": (self.query_stats["successful_queries"] / 
                           self.query_stats["total_queries"] * 100 if self.query_stats["total_queries"] > 0 else 0),
            "total_retrieved_documents": self.query_stats["total_retrieved"],
            "average_retrieval_time": avg_time,
            "total_query_time": self.query_stats["total_time"],
            "cache_size": self.query_cache.size(),
            "cache_hits": self.query_stats["cache_hits"],
            "cache_misses": self.query_stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "query_history_size": len(self.query_history),
            "performance_metrics_count": len(self.performance_metrics)
        }
    
    def clear_cache(self) -> None:
        """清空查询缓存
"""
        self.query_cache.clear()
        self.query_stats["cache_hits"] = 0
        self.query_stats["cache_misses"] = 0
        logger.info("查询缓存已清空")
    
    def export_query_history(self, filepath: str = "query_history.json") -> None:
        """导出查询历史
"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.query_history, f, ensure_ascii=False, indent=2)
            logger.info(f"查询历史已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出查询历史失败: {e}")
    
    def export_performance_metrics(self, filepath: str = "performance_metrics.json") -> None:
        """导出性能指标
"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.performance_metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"性能指标已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出性能指标失败: {e}")
    
    def __del__(self):
        """析构函数，清理资源
"""
        logger.info("VectorDBQuery实例销毁，清理资源...")
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)