#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量数据库查询器
================================================================
- 城市、区名及映射从外部 JSON 文件加载，无需修改代码即可扩展。
- 查询分类逻辑移至主类，缓存类不再依赖硬编码数据。
- 保持向后兼容：若未提供配置文件或加载失败，回退至内置默认数据。
- 类名 OptimizedVectorDBQuery 已重命名为 RetrievalEngine，并添加 RetrievalConfig。
================================================================
"""

import os
import json
import time
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import re
import jieba
import jieba.analyse
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ===================== 检索配置类 =====================
@dataclass
class RetrievalConfig:
    """检索配置类，与 RetrievalEngine 的初始化参数对应"""
    chroma_db_path: str = "/tmp/chroma_db_dsw"
    embedding_model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
    collection_name: str = "rag_knowledge_base"
    device: Optional[str] = None
    rerank_enabled: bool = False
    # 可根据需要添加更多配置项（如性能模式等）


# ===================== 重排序器基类与实现 =====================
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        pass

class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        actual_model = model_path if model_path else model_name
        logger.info(f"加载重排序模型: {actual_model} | 设备: {self.device}")
        self.model = CrossEncoder(
            actual_model,
            device=self.device,
            trust_remote_code=True,
            local_files_only=True,
        )
        logger.info("重排序模型加载完成")

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        pairs = [(query, doc["content"]) for doc in docs]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        for doc in reranked[:top_k]:
            doc["similarity"] = round(doc["rerank_score"], 4)
            doc["retrieval_method"] += "+rerank"
            doc["reranked"] = True
        return reranked[:top_k]

# ===================== 默认配置（可覆盖） =====================
DEFAULT_CONFIG = {
    "PERFORMANCE_MODE": "balanced",
    "cache_max_size": 2000,
    "cache_expire_time": 3600,
    "cache_clean_interval": 300,
    "enable_query_cache": True,
    "enable_vector_cache": True,
    "vector_cache_ttl": 1800,
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "credit_code_exact_match_score": 1.0,
    "credit_code_id_match_score": 0.95,
    "rerank_enabled_by_default": False,
    "rerank_candidate_multiple": 3,
    "device": None,
    "industry_terms_path": "./industry_terms.txt",
    "product_terms_path": "./product_terms.txt",
    "legal_terms_path": "./legal_terms.txt",
    "address_terms_path": "./address_terms.txt",
    "business_terms_path": "./business_terms.txt",
    "bm25_index_enabled": True,
    "bm25_index_refresh_interval": 3600,
    "bm25_index_max_docs": 50000,
    "hnsw_M": 16,
    "hnsw_construction_ef": 100,
    "hnsw_search_ef": 40,
}

# ===================== 内置默认城市数据（作为加载失败时的回退） =====================
DEFAULT_COMMON_CITIES = [
    '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '重庆',
    '天津', '苏州', '西安', '郑州', '长沙', '沈阳', '青岛', '宁波', '东莞',
    '无锡', '佛山', '合肥', '福州', '泉州', '济南', '哈尔滨', '长春', '大连',
    '昆明', '厦门', '太原', '南昌', '南宁', '贵阳', '海口', '兰州', '西宁',
    '呼和浩特', '乌鲁木齐', '拉萨', '银川', '香港', '澳门', '台北'
]

DEFAULT_COMMON_DISTRICTS = [
    '松江', '浦东', '静安', '黄浦', '徐汇', '长宁', '普陀', '虹口', '杨浦', '闵行',
    '宝山', '嘉定', '金山', '青浦', '奉贤', '崇明', '南山', '福田', '罗湖', '龙岗',
    '宝安', '龙华', '坪山', '光明', '盐田', '天河', '越秀', '海珠', '荔湾', '白云',
    '黄埔', '番禺', '花都', '南沙', '从化', '增城', '江宁', '鼓楼', '玄武', '建邺',
    '秦淮', '栖霞', '雨花台', '浦口', '六合', '溧水', '高淳', '西湖', '上城', '拱墅',
    '滨江', '萧山', '余杭', '富阳', '临安', '桐庐', '淳安', '建德'
]

DEFAULT_CITY_TO_DISTRICTS = {
    '北京': ['东城', '西城', '朝阳', '丰台', '石景山', '海淀', '门头沟', '房山', '通州', '顺义', '昌平', '大兴', '怀柔', '平谷', '密云', '延庆'],
    '上海': ['黄浦', '徐汇', '长宁', '静安', '普陀', '虹口', '杨浦', '闵行', '宝山', '嘉定', '浦东', '金山', '松江', '青浦', '奉贤', '崇明'],
    '广州': ['荔湾', '越秀', '海珠', '天河', '白云', '黄埔', '番禺', '花都', '南沙', '从化', '增城'],
    '深圳': ['罗湖', '福田', '南山', '宝安', '龙岗', '盐田', '龙华', '坪山', '光明'],
    '杭州': ['上城', '下城', '江干', '拱墅', '西湖', '滨江', '萧山', '余杭', '富阳', '临安', '桐庐', '淳安', '建德'],
    '南京': ['玄武', '秦淮', '建邺', '鼓楼', '浦口', '栖霞', '雨花台', '江宁', '六合', '溧水', '高淳'],
    '武汉': ['江岸', '江汉', '硚口', '汉阳', '武昌', '青山', '洪山', '东西湖', '汉南', '蔡甸', '江夏', '黄陂', '新洲'],
    '成都': ['锦江', '青羊', '金牛', '武侯', '成华', '龙泉驿', '青白江', '新都', '温江', '双流', '郫都', '金堂', '大邑', '蒲江', '新津'],
    '重庆': ['万州', '涪陵', '渝中', '大渡口', '江北', '沙坪坝', '九龙坡', '南岸', '北碚', '綦江', '大足', '渝北', '巴南', '黔江', '长寿', '江津', '合川', '永川', '南川', '璧山', '铜梁', '潼南', '荣昌', '开州', '梁平', '武隆'],
    '天津': ['和平', '河东', '河西', '南开', '河北', '红桥', '东丽', '西青', '津南', '北辰', '武清', '宝坻', '滨海', '宁河', '静海', '蓟州'],
    '苏州': ['虎丘', '吴中', '相城', '姑苏', '吴江', '常熟', '张家港', '昆山', '太仓'],
    '西安': ['新城', '碑林', '莲湖', '灞桥', '未央', '雁塔', '阎良', '临潼', '长安', '高陵', '鄠邑', '蓝田', '周至'],
    '郑州': ['中原', '二七', '管城', '金水', '上街', '惠济', '中牟', '巩义', '荥阳', '新密', '新郑', '登封'],
    '长沙': ['芙蓉', '天心', '岳麓', '开福', '雨花', '望城', '长沙县', '宁乡', '浏阳'],
    '沈阳': ['和平', '沈河', '大东', '皇姑', '铁西', '苏家屯', '浑南', '沈北', '于洪', '辽中', '康平', '法库', '新民'],
    '青岛': ['市南', '市北', '黄岛', '崂山', '李沧', '城阳', '即墨', '胶州', '平度', '莱西'],
    '宁波': ['海曙', '江北', '北仑', '镇海', '鄞州', '奉化', '象山', '宁海', '余姚', '慈溪'],
    '东莞': ['莞城', '南城', '万江', '东城', '石碣', '石龙', '茶山', '石排', '企石', '横沥', '桥头', '谢岗', '东坑', '常平', '寮步', '樟木头', '大朗', '黄江', '清溪', '塘厦', '凤岗', '大岭山', '长安', '虎门', '厚街', '沙田', '道滘', '洪梅', '麻涌', '望牛墩', '中堂', '高埗'],
    '无锡': ['锡山', '惠山', '滨湖', '梁溪', '新吴', '江阴', '宜兴'],
    '佛山': ['禅城', '南海', '顺德', '三水', '高明'],
    '合肥': ['瑶海', '庐阳', '蜀山', '包河', '长丰', '肥东', '肥西', '庐江', '巢湖'],
    '福州': ['鼓楼', '台江', '仓山', '马尾', '晋安', '长乐', '闽侯', '连江', '罗源', '闽清', '永泰', '平潭', '福清'],
    '泉州': ['鲤城', '丰泽', '洛江', '泉港', '石狮', '晋江', '南安', '惠安', '安溪', '永春', '德化', '金门'],
    '济南': ['历下', '市中', '槐荫', '天桥', '历城', '长清', '章丘', '济阳', '莱芜', '钢城', '平阴', '商河'],
    '哈尔滨': ['道里', '南岗', '道外', '平房', '松北', '香坊', '呼兰', '阿城', '双城', '依兰', '方正', '宾县', '巴彦', '木兰', '通河', '延寿', '尚志', '五常'],
    '长春': ['南关', '宽城', '朝阳', '二道', '绿园', '双阳', '九台', '农安', '榆树', '德惠'],
    '大连': ['中山', '西岗', '沙河口', '甘井子', '旅顺口', '金州', '普兰店', '长海', '瓦房店', '庄河'],
    '昆明': ['五华', '盘龙', '官渡', '西山', '东川', '呈贡', '晋宁', '富民', '宜良', '石林', '嵩明', '禄劝', '寻甸', '安宁'],
    '厦门': ['思明', '海沧', '湖里', '集美', '同安', '翔安'],
    '太原': ['小店', '迎泽', '杏花岭', '尖草坪', '万柏林', '晋源', '清徐', '阳曲', '娄烦', '古交'],
    '南昌': ['东湖', '西湖', '青云谱', '湾里', '青山湖', '新建', '南昌县', '安义', '进贤'],
    '南宁': ['兴宁', '青秀', '江南', '西乡塘', '良庆', '邕宁', '武鸣', '隆安', '马山', '上林', '宾阳', '横州'],
    '贵阳': ['南明', '云岩', '花溪', '乌当', '白云', '观山湖', '开阳', '息烽', '修文', '清镇'],
    '海口': ['秀英', '龙华', '琼山', '美兰'],
    '兰州': ['城关', '七里河', '西固', '安宁', '红古', '永登', '皋兰', '榆中'],
    '西宁': ['城东', '城中', '城西', '城北', '湟中', '湟源', '大通'],
    '呼和浩特': ['新城', '回民', '玉泉', '赛罕', '土默特左', '托克托', '和林格尔', '清水河', '武川'],
    '乌鲁木齐': ['天山', '沙依巴克', '新市', '水磨沟', '头屯河', '达坂城', '米东', '乌鲁木齐县'],
    '拉萨': ['城关', '堆龙德庆', '达孜', '林周', '当雄', '尼木', '曲水', '墨竹工卡'],
    '银川': ['兴庆', '西夏', '金凤', '永宁', '贺兰', '灵武'],
    '香港': ['中西', '湾仔', '东区', '南区', '油尖旺', '深水埗', '九龙城', '黄大仙', '观塘', '荃湾', '屯门', '元朗', '北区', '大埔', '沙田', '西贡', '离岛'],
    '澳门': ['花地玛', '圣安多', '大堂', '望德', '风顺'],
    '台北': ['中正', '大同', '中山', '松山', '大安', '万华', '信义', '士林', '北投', '内湖', '南港', '文山'],
    '台州': ['椒江', '黄岩', '路桥', '临海', '温岭', '玉环', '天台', '仙居', '三门']
}

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

# ===================== 线程安全缓存系统（不再依赖硬编码分类） =====================
class ThreadSafeQueryCache:
    def __init__(self, config: dict):
        self.config = config
        self.max_size = config["cache_max_size"]
        self.expire_time = config["cache_expire_time"]
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
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry["timestamp"] < self.expire_time:
                    self.stats["hits"] += 1
                    self.cache.move_to_end(key)
                    return entry["result"]
                else:
                    del self.cache[key]
                    self.stats["expired_evictions"] += 1
            self.stats["misses"] += 1
            return None

    def set(self, query: str, params: Dict, result: Any) -> None:
        with self.lock:
            key = self._generate_key(query, params)
            self.cache[key] = {
                "result": result,
                "timestamp": time.time(),
                "query": query[:100],
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
        if not self.config["enable_vector_cache"]:
            return None
        with self.lock:
            key = self._simple_key(text)
            if key in self.vector_cache:
                entry = self.vector_cache[key]
                if time.time() - entry["timestamp"] < self.config["vector_cache_ttl"]:
                    self.stats["vector_cache_hits"] += 1
                    self.vector_cache.move_to_end(key)
                    return entry["vector"]
                else:
                    del self.vector_cache[key]
                    self.stats["expired_evictions"] += 1
            self.stats["vector_cache_misses"] += 1
            return None

    def set_vector(self, text: str, vector: np.ndarray) -> None:
        if not self.config["enable_vector_cache"]:
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
                           if now - v["timestamp"] >= self.config["vector_cache_ttl"]]
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
            time.sleep(self.config["cache_clean_interval"])
            try:
                self.remove_expired()
                logger.debug("定时清理过期缓存完成")
            except Exception as e:
                logger.error(f"定时清理缓存失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            stats = self.stats.copy()
            total = stats["total_requests"]
            stats["hit_rate"] = stats["hits"] / total if total > 0 else 0.0
            stats["uptime"] = time.time() - stats["last_clear_time"]
            return stats

# ===================== 主查询类（已重命名为 RetrievalEngine） =====================
class RetrievalEngine:
    """
    检索引擎（原 OptimizedVectorDBQuery）
    支持向量检索、混合检索、重排序、城市映射等功能。
    """
    def __init__(self,
                 db_path: str = "/tmp/chroma_db_dsw",
                 model_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 collection_name: str = "rag_knowledge_base",
                 device: str = None,
                 reranker: Optional[BaseReranker] = None,
                 config: Optional[dict] = None,
                 location_config_path: Optional[str] = None):
        """
        初始化检索引擎
        :param db_path: ChromaDB 持久化路径
        :param model_path: Embedding 模型路径
        :param collection_name: 集合名称
        :param device: 设备（cuda/cpu）
        :param reranker: 重排序器实例
        :param config: 配置字典，若不提供则使用 DEFAULT_CONFIG
        :param location_config_path: 城市配置文件路径（JSON），若不提供或加载失败则使用内置默认数据
        """
        self.db_path = db_path
        self.model_path = model_path
        self.collection_name = collection_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker = reranker

        # 合并配置：优先使用传入的 config，否则使用默认，并基于 PERFORMANCE_MODE 计算派生配置
        base_config = DEFAULT_CONFIG.copy()
        if config:
            base_config.update(config)
        self.config = base_config
        self._update_performance_mode()

        # 加载城市数据
        self.common_cities, self.common_districts, self.city_to_districts = self._load_location_config(location_config_path)

        logger.info(f"初始化检索增强引擎 | 设备: {self.device}")
        if self.reranker:
            logger.info(f"已启用重排序器: {self.reranker.__class__.__name__}")

        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.query_cache = ThreadSafeQueryCache(self.config)

        self.industry_terms = self._load_terms_dict(self.config["industry_terms_path"],
                                                   default=['建筑', '装饰', '装修', '工程', '设计', '施工'])
        self.product_terms = self._load_terms_dict(self.config["product_terms_path"],
                                                  default=['设备', '仪器', '装置', '机器', '工具', '材料'])
        self.legal_terms = self._load_terms_dict(self.config["legal_terms_path"],
                                                default=['法律', '法规', '条例', '规定', '办法', '司法解释'])
        self.address_terms = self._load_terms_dict(self.config["address_terms_path"],
                                                  default=['省', '市', '区', '县', '镇', '街道', '路', '巷', '弄', '号'])
        self.business_terms = self._load_terms_dict(self.config["business_terms_path"],
                                                   default=['经营', '业务', '销售', '制造', '生产', '服务', '加工', '贸易'])

        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_doc_ids = []
        self.bm25_doc_id_to_idx = {}
        self.bm25_last_refresh = 0
        self.bm25_lock = threading.Lock()
        self._query_tokens_cache = {}
        self._query_tokens_lock = threading.Lock()

        self._internal_config = {
            "default_top_k": 20,
            "enable_cache": self.config["enable_query_cache"],
            "enable_hybrid_search": True,
            "min_similarity_default": 0.0,
            "bm25_index_enabled": self.config["bm25_index_enabled"],
        }

        self._initialize_components()

    def _load_location_config(self, path: Optional[str]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        从 JSON 文件加载城市、区名及映射关系
        若文件不存在或加载失败，返回内置默认数据
        """
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                common_cities = data.get("common_cities", DEFAULT_COMMON_CITIES)
                common_districts = data.get("common_districts", DEFAULT_COMMON_DISTRICTS)
                city_to_districts = data.get("city_to_districts", DEFAULT_CITY_TO_DISTRICTS)
                logger.info(f"成功加载城市配置文件: {path}")
                return common_cities, common_districts, city_to_districts
            except Exception as e:
                logger.warning(f"加载城市配置文件失败: {e}，将使用内置默认数据")
        else:
            if path:
                logger.warning(f"城市配置文件不存在: {path}，将使用内置默认数据")
        return DEFAULT_COMMON_CITIES, DEFAULT_COMMON_DISTRICTS, DEFAULT_CITY_TO_DISTRICTS

    def _update_performance_mode(self):
        """根据 PERFORMANCE_MODE 更新派生配置"""
        mode = self.config["PERFORMANCE_MODE"]
        if mode == "balanced":
            self.config["hybrid_search_candidate_multiple"] = 8
            self.config["credit_code_candidate_multiple"] = 8
            self.config["force_recreate_collection"] = False
            logger.info("性能模式: balanced (8倍候选，召回最优)")
        elif mode == "fast":
            self.config["hybrid_search_candidate_multiple"] = 6
            self.config["credit_code_candidate_multiple"] = 6
            self.config["force_recreate_collection"] = False
            logger.info("性能模式: fast (6倍候选，召回略降，速度提升25%)")
        elif mode == "ultra":
            self.config["hybrid_search_candidate_multiple"] = 7
            self.config["credit_code_candidate_multiple"] = 7
            self.config["hnsw_search_ef"] = 80                   # 从40提升到80
            self.config["force_recreate_collection"] = True   # ultra 模式下强制重建集合以应用 HNSW 优化
            logger.info("性能模式: ultra (7倍候选 + HNSW优化，速度提升50%，需重建集合)")
        else:
            raise ValueError(f"未知的性能模式: {mode}，可选 balanced/fast/ultra")

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

            if self.config["force_recreate_collection"]:
                try:
                    chroma_client.delete_collection(name=self.collection_name)
                    logger.warning(f"已删除集合 {self.collection_name}（性能模式 ultra 重建）")
                except:
                    pass

            try:
                collection = chroma_client.get_collection(name=self.collection_name)
                logger.info(f"使用现有集合 {self.collection_name}")
            except Exception:
                if self.config["PERFORMANCE_MODE"] == "ultra":
                    hnsw_metadata = {
                        "hnsw:space": "cosine",
                        "hnsw:M": self.config["hnsw_M"],
                        "hnsw:construction_ef": self.config["hnsw_construction_ef"],
                        "hnsw:search_ef": self.config["hnsw_search_ef"],
                    }
                    logger.info(f"应用 HNSW 优化参数: {hnsw_metadata}")
                else:
                    hnsw_metadata = {"hnsw:space": "cosine"}

                collection = chroma_client.create_collection(
                    name=self.collection_name,
                    metadata=hnsw_metadata
                )
                logger.info(f"创建新集合 {self.collection_name}")

            self.embedding_model = embedding_model
            self.chroma_client = chroma_client
            self.collection = collection

            if self._internal_config["bm25_index_enabled"]:
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
        if not self._internal_config["bm25_index_enabled"] or not self.collection:
            return False
        with self.bm25_lock:
            now = time.time()
            if not force and (now - self.bm25_last_refresh) < self.config["bm25_index_refresh_interval"]:
                return False
            try:
                logger.info("开始刷新 BM25 索引...")
                all_docs = self.collection.get(
                    limit=self.config["bm25_index_max_docs"],
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

    def _prefilter(self, docs: List[Dict], query_type: str, query_text: str) -> List[Dict]:
        if not docs:
            return docs

        filtered = docs
        if query_type == "address":
            address_keywords = ['省', '市', '区', '县', '镇', '街道', '路', '巷', '弄', '号', '村']
            filtered = [d for d in docs if any(kw in d.get('content', '') for kw in address_keywords)]
        elif query_type == "legal_representative":
            import re
            names = re.findall(r'[\u4e00-\u9fa5]{2,4}', query_text)
            if names:
                filtered = [d for d in docs if any(name in d.get('content', '') for name in names)]
        elif query_type in ("product_keyword", "product_natural"):
            words = [word for word, flag in pseg.cut(query_text) if flag.startswith('n')]
            required_terms = set(self.product_terms) | set(words)
            filtered = [d for d in docs if any(term in d.get('content', '') for term in required_terms)]

        if not filtered:
            logger.debug(f"预过滤后结果为空，回退至原始列表 | 类型: {query_type}")
            return docs
        return filtered

    def _extract_places(self, query_text: str) -> List[str]:
        """从查询中提取所有可能的地名（城市或区）"""
        places = []
        for city in self.common_cities:
            if city in query_text:
                places.append(city)
        for dist in self.common_districts:
            if dist in query_text:
                places.append(dist)
        return list(set(places))

    def _expand_city_to_districts(self, city: str) -> List[str]:
        """将城市名扩展为城市本身 + 所辖区名列表"""
        places = [city]  # 始终包含城市名本身
        if city in self.city_to_districts:
            places.extend(self.city_to_districts[city])
        return list(set(places))

    def _classify_query(self, query: str) -> str:
        """查询分类，使用实例中的城市数据"""
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
        city_count = sum(1 for city in self.common_cities if city in query)
        district_count = sum(1 for dist in self.common_districts if dist in query)
        total_places = city_count + district_count
        if total_places >= 2:
            return "cross_city"
        if total_places == 1 and any(kw in q for kw in ['地址', '注册地', '所在地', '位于', '在']):
            return "address"
        return "general"

    def query(self,
              query_text: str,
              top_k: int = None,
              where_filter: Optional[Dict[str, Any]] = None,
              min_similarity: Optional[float] = None,
              return_format: str = "structured",
              use_rerank: bool = None,
              **kwargs) -> Union[QueryResult, List[Dict[str, Any]]]:
        start_time = time.time()
        top_k = top_k or self._internal_config["default_top_k"]
        min_similarity = min_similarity or self._internal_config["min_similarity_default"]
        if not isinstance(query_text, str) or not query_text.strip():
            logger.error("查询文本为空")
            empty = QueryResult(query=query_text, retrieved_documents=[], retrieval_time=0,
                               total_retrieved=0, avg_similarity=0.0, query_type="general")
            return empty if return_format == "structured" else []

        query_type = self._classify_query(query_text)   # 使用实例方法

        if use_rerank is None:
            use_rerank = self.config["rerank_enabled_by_default"] and self.reranker is not None
        else:
            use_rerank = use_rerank and self.reranker is not None

        cache_params = {
            "top_k": top_k,
            "where_filter": where_filter,
            "min_similarity": min_similarity,
            "query_type": query_type,
            "use_rerank": use_rerank,
            "rerank_multiple": self.config["rerank_candidate_multiple"] if use_rerank else None,
        }

        if self._internal_config["enable_cache"]:
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
                final_docs = docs[:top_k]
                reranked = False
            else:
                query_to_search = query_text

                if use_rerank:
                    type_multiple_map = {
                        "company_name": 2,
                        "credit_code": 1,
                        "address": 2,
                        "business_scope": 2,
                        "legal_representative": 3,
                        "product_name": 3,
                        "product_supplier": 3,
                        "product_keyword": 3,
                        "product_natural": 3,
                        "cross_city": 8,
                        "general": 3,
                    }
                    multiple = type_multiple_map.get(query_type, self.config["rerank_candidate_multiple"])
                    candidate_k = top_k * multiple
                    logger.debug(f"启用重排序，候选倍数: {multiple} (类型: {query_type}), 候选集大小: {candidate_k}")
                else:
                    candidate_k = top_k * self.config["hybrid_search_candidate_multiple"]

                places_in_query = self._extract_places(query_text)

                # 智能过滤：仅对明确地址查询且提取到地名时应用 city 过滤
                apply_city_filter = False
                place_filter = None
                if places_in_query:
                    is_address_like = any(kw in query_text for kw in ['地址', '注册地', '所在地', '位于', '在'])
                    if query_type in ("address", "cross_city") or is_address_like:
                        # 对 cross_city 类型，将城市名扩展为所辖区名列表
                        expanded_places = []
                        for p in places_in_query:
                            if p in self.common_cities:
                                expanded_places.extend(self._expand_city_to_districts(p))
                            else:
                                expanded_places.append(p)
                        expanded_places = list(set(expanded_places))
                        place_filter = {"city": {"$in": expanded_places}}
                        apply_city_filter = True
                        logger.debug(f"将应用城市过滤，扩展后地名: {expanded_places}")

                # 先尝试无过滤的检索，获取足够的候选
                docs = self._hybrid_search(query_to_search, candidate_k, where_filter, min_similarity)

                # 如果启用过滤且结果数少于 top_k，则放宽条件（取消过滤）
                if apply_city_filter and docs:
                    # 应用过滤
                    filtered_docs = [d for d in docs if d.get("metadata", {}).get("city") in expanded_places]
                    if len(filtered_docs) < top_k:
                        logger.debug(f"过滤后结果数 {len(filtered_docs)} < top_k {top_k}，回退到无过滤")
                    else:
                        docs = filtered_docs
                        # 更新 where_filter 以备后续使用（但已无用）
                        if place_filter:
                            if where_filter:
                                where_filter = {"$and": [where_filter, place_filter]}
                            else:
                                where_filter = place_filter

                if not docs:
                    final_docs = []
                    method = "hybrid_search"
                    reranked = False
                else:
                    method = "hybrid_search"
                    reranked = False
                    if use_rerank:
                        docs = self._prefilter(docs, query_type, query_text)
                    if use_rerank and self.reranker is not None:
                        docs = self.reranker.rerank(query_to_search, docs, top_k)
                        method += "+rerank"
                        reranked = True
                        final_docs = docs[:top_k]
                    else:
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
                reranked=reranked,
                query_type=query_type
            )

            if self._internal_config["enable_cache"]:
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

    def _hybrid_search(self, query_text: str, top_k: int,
                       where_filter: Optional[Dict] = None,
                       min_similarity: Optional[float] = None) -> List[Dict]:
        candidate_k = top_k * self.config["hybrid_search_candidate_multiple"]
        vec_docs = self._vector_search_optimized(query_text, candidate_k, where_filter, 0.0)
        if not vec_docs:
            return []

        with self._query_tokens_lock:
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
            hybrid = vec_score * self.config["vector_weight"] + bm25_norm * self.config["bm25_weight"]
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
                doc["similarity"] = self.config["credit_code_exact_match_score"]
                doc["exact_match"] = True
                exact_matched.append(doc)
        if exact_matched:
            return exact_matched[:top_k]
        candidates = self._vector_search_optimized(
            credit_code,
            top_k * self.config["credit_code_candidate_multiple"],
            where_filter, 0.0
        )
        return candidates[:top_k]

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
            },
            "reranker_loaded": self.reranker is not None,
            "location_data_loaded": {
                "common_cities": len(self.common_cities),
                "common_districts": len(self.common_districts),
                "city_to_districts": len(self.city_to_districts)
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
        with self._query_tokens_lock:
            self._query_tokens_cache.clear()
        logger.info("资源释放完成")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    # 示例用法（可传入城市配置文件路径）
    engine = RetrievalEngine(
        db_path="/tmp/chroma_db_dsw",
        model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
        collection_name="rag_knowledge_base",
        location_config_path="location_config.json"  # 假设配置文件在当前目录
    )
    try:
        status = engine.test_connection()
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
            result = engine.query(q, top_k=3)
            print(f"类型: {result.query_type}, 方法: {result.retrieval_method}, 耗时: {result.retrieval_time:.3f}s")
            for i, doc in enumerate(result.retrieved_documents):
                print(f"  [{i+1}] {doc['similarity']:.4f} - {doc['content_preview'][:80]}...")
    finally:
        engine.close()