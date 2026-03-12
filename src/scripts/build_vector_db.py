#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSW环境专用RAG向量数据库构建脚本
使用容器本地存储避免OSS I/O问题
"""

import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import argparse
import shutil
import hashlib
import time
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DSWVectorDatabaseBuilder:
    """DSW环境专用向量数据库构建器"""
    
    def __init__(self, 
                 knowledge_base_path: str,
                 embedding_local_path: str = "/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                 llm_local_path: str = "/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct",
                 chroma_persist_dir: str = "/tmp/chroma_db_dsw",  # 使用/tmp目录
                 device: str = None):
        """
        初始化向量数据库构建器
        
        Args:
            knowledge_base_path: 知识库JSON文件路径（可以在OSS中）
            embedding_local_path: embedding模型路径
            llm_local_path: 大语言模型路径
            chroma_persist_dir: ChromaDB持久化目录（使用容器本地存储）
            device: 运行设备
        """
        self.knowledge_base_path = knowledge_base_path
        self.embedding_local_path = embedding_local_path
        self.llm_local_path = llm_local_path
        self.chroma_persist_dir = chroma_persist_dir
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"Embedding模型路径: {self.embedding_local_path}")
        logger.info(f"ChromaDB目录（本地存储）: {self.chroma_persist_dir}")
        
        # 确保使用本地存储目录
        if self.chroma_persist_dir.startswith('/mnt/workspace'):
            logger.warning(f"警告：不建议将ChromaDB放在OSS挂载目录 {self.chroma_persist_dir}")
            logger.warning("建议使用 /tmp 或 /home 目录")
        
        # 检查模型路径是否存在
        self.check_model_paths()
        
        # 初始化模型
        self.embedding_model = None
        self.llm_model = None
        self.tokenizer = None
        self.chroma_client = None
        self.collection = None
        
        # 加载数据
        self.documents = self.load_knowledge_base()
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "embedding_dimension": 0
        }
        
        # ID跟踪器
        self.id_counter = {}
        self.duplicate_content_tracker = {}  # 用于跟踪重复内容
    
    def check_model_paths(self):
        """检查模型路径"""
        logger.info("检查模型路径...")
        
        # 检查embedding模型
        if os.path.exists(self.embedding_local_path):
            logger.info(f"✅ Embedding模型存在: {self.embedding_local_path}")
        else:
            logger.warning(f"⚠️  Embedding模型不存在: {self.embedding_local_path}")
            # 尝试从Modelscope下载
            try:
                import modelscope
                from modelscope import snapshot_download
                logger.info("尝试从ModelScope下载BGE模型...")
                self.embedding_local_path = snapshot_download("BAAI/bge-m3", cache_dir="/mnt/workspace/data/modelscope/cache")
            except ImportError:
                logger.error("modelscope库未安装，请安装: pip install modelscope")
                raise
        
        # 检查LLM模型
        if os.path.exists(self.llm_local_path):
            logger.info(f"✅ LLM模型存在: {self.llm_local_path}")
        else:
            logger.warning(f"⚠️  LLM模型不存在: {self.llm_local_path}")
            # 尝试从Modelscope下载
            try:
                logger.info("尝试从ModelScope下载Qwen模型...")
                self.llm_local_path = snapshot_download("qwen/Qwen2.5-3B-Instruct", cache_dir="/mnt/workspace/data/modelscope/cache")
            except Exception as e:
                logger.warning(f"下载Qwen模型失败: {e}")
    
    def load_knowledge_base(self) -> List[Dict[str, Any]]:
        """加载知识库数据"""
        try:
            # 检查知识库文件是否存在
            if not os.path.exists(self.knowledge_base_path):
                logger.error(f"知识库文件不存在: {self.knowledge_base_path}")
                # 尝试在常见位置查找
                possible_paths = [
                    self.knowledge_base_path,
                    f"/mnt/workspace/{self.knowledge_base_path}",
                    f"./{self.knowledge_base_path}",
                    "/mnt/workspace/data/rag_knowledge_base_20260208_224240/knowledge_base.json"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"找到知识库文件: {path}")
                        self.knowledge_base_path = path
                        break
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data.get("documents", [])
            logger.info(f"成功加载 {len(documents)} 个文档")
            
            # 显示统计信息
            stats = data.get("stats", {})
            total_docs = stats.get('total_documents', 0)
            logger.info(f"文档总数: {total_docs}")
            
            # 显示前几个文档类型
            type_stats = stats.get('documents_by_type', {})
            for doc_type, count in list(type_stats.items())[:5]:
                logger.info(f"  - {doc_type}: {count} 个文档")
            
            return documents
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            logger.info(f"尝试的文件路径: {self.knowledge_base_path}")
            
            # 列出可能的文件
            logger.info("查找当前目录下的文件:")
            os.system("ls -la | grep -i json || echo '没有找到JSON文件'")
            
            return []
    
    def initialize_embedding_model(self):
        """初始化embedding模型"""
        logger.info(f"正在加载embedding模型...")
        
        try:
            # 使用sentence-transformers加载BGE模型
            self.embedding_model = SentenceTransformer(
                self.embedding_local_path,
                device=self.device
            )
            
            # 测试模型
            test_embeddings = self.embedding_model.encode(["测试文本"])
            embedding_dim = test_embeddings.shape[1]
            
            logger.info(f"✅ Embedding模型加载成功")
            logger.info(f"   向量维度: {embedding_dim}")
            
            self.stats["embedding_dimension"] = embedding_dim
            
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            
            # 尝试从HuggingFace下载
            logger.info("尝试从HuggingFace下载模型...")
            try:
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-m3",
                    device=self.device
                )
                logger.info("✅ 从HuggingFace下载成功")
            except Exception as e2:
                logger.error(f"从HuggingFace下载也失败: {e2}")
                raise
    
    def initialize_chroma_db(self):
        """初始化ChromaDB（使用本地存储）"""
        logger.info(f"初始化ChromaDB，目录: {self.chroma_persist_dir}")
        
        try:
            # 确保目录存在
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            
            # 如果目录权限有问题，尝试使用/tmp子目录
            if not os.access(self.chroma_persist_dir, os.W_OK):
                logger.warning(f"目录 {self.chroma_persist_dir} 不可写")
                self.chroma_persist_dir = f"/tmp/chroma_db_{os.getpid()}"
                os.makedirs(self.chroma_persist_dir, exist_ok=True)
                logger.info(f"切换到目录: {self.chroma_persist_dir}")
            
            # 创建ChromaDB客户端
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # 创建或获取集合
            collection_name = "rag_knowledge_base"
            
            # 检查集合是否存在
            try:
                existing_collections = [col.name for col in self.chroma_client.list_collections()]
            except:
                existing_collections = []
            
            if collection_name in existing_collections:
                logger.warning(f"集合 '{collection_name}' 已存在，将删除重建")
                self.chroma_client.delete_collection(collection_name)
            
            # 创建新的集合
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "RAG知识库",
                    "embedding_model": "BAAI/bge-m3",
                    "llm_model": "Qwen2.5-3B-Instruct",
                    "source": self.knowledge_base_path,
                    "build_env": "阿里云PAI DSW"
                }
            )
            
            logger.info(f"✅ ChromaDB集合 '{collection_name}' 创建成功")
            logger.info(f"   存储位置: {self.chroma_persist_dir}")
            
        except Exception as e:
            logger.error(f"初始化ChromaDB失败: {e}")
            
            # 尝试备用方案：使用临时目录
            logger.info("尝试使用备用方案...")
            try:
                temp_dir = f"/tmp/chroma_temp_{os.getpid()}"
                os.makedirs(temp_dir, exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(
                    path=temp_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                self.collection = self.chroma_client.create_collection(
                    name="rag_knowledge_base"
                )
                
                self.chroma_persist_dir = temp_dir
                logger.info(f"✅ 使用临时目录成功: {temp_dir}")
                
            except Exception as e2:
                logger.error(f"备用方案也失败: {e2}")
                raise
    
    def generate_unique_id(self, base_id: str) -> str:
        """生成唯一ID，避免重复"""
        if base_id not in self.id_counter:
            self.id_counter[base_id] = 0
            return base_id
        else:
            self.id_counter[base_id] += 1
            return f"{base_id}_{self.id_counter[base_id]}"
    
    def generate_content_hash_id(self, content: str, doc_id: str = "", chunk_index: int = 0) -> str:
        """基于内容生成哈希ID，确保内容相同生成相同ID"""
        # 创建内容哈希
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
        
        # 如果提供了doc_id，结合doc_id和内容哈希
        if doc_id:
            return f"{doc_id}_{content_hash}"
        else:
            return content_hash
    
    def chunk_documents(self, max_chunk_size: int = 500) -> List[Dict[str, Any]]:
        """将文档分块"""
        chunked_docs = []
        id_tracker = set()  # 跟踪已使用的ID
        duplicate_counter = 0
        
        logger.info(f"开始文档分块处理，最大块大小: {max_chunk_size}")
        
        for doc_idx, doc in enumerate(tqdm(self.documents, desc="文档分块", unit="doc")):
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", "")
            doc_type = doc.get("type", "unknown")
            
            if not content:
                continue
            
            # 确保基础doc_id不为空
            if not doc_id:
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{doc_idx}"
            
            # 如果文档较短，直接使用
            if len(content) <= max_chunk_size:
                # 生成基于内容的哈希ID
                chunk_id = self.generate_content_hash_id(content, doc_id)
                
                # 检查这个ID是否已经存在
                if chunk_id in id_tracker:
                    duplicate_counter += 1
                    logger.debug(f"发现重复ID: {chunk_id}")
                    
                    # 如果是内容完全相同，我们可能需要记录但跳过
                    # 这里我们生成一个新的UUID来避免重复
                    chunk_id = str(uuid.uuid4())
                
                id_tracker.add(chunk_id)
                chunked_docs.append({
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "type": doc_type,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "original_doc_id": doc_id,
                        "content_hash": hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
                    },
                    "type": doc_type
                })
            else:
                # 长文档需要分块
                chunks = self.split_text(content, max_chunk_size)
                for i, chunk in enumerate(chunks):
                    # 生成基于内容的哈希ID
                    chunk_id = self.generate_content_hash_id(chunk, doc_id, i)
                    
                    # 检查这个ID是否已经存在
                    if chunk_id in id_tracker:
                        duplicate_counter += 1
                        logger.debug(f"发现重复ID: {chunk_id}")
                        
                        # 如果是内容完全相同，我们可能需要记录但跳过
                        # 这里我们生成一个新的UUID来避免重复
                        chunk_id = str(uuid.uuid4())
                    
                    id_tracker.add(chunk_id)
                    chunked_docs.append({
                        "id": chunk_id,
                        "content": chunk,
                        "metadata": {
                            **doc.get("metadata", {}),
                            "type": doc_type,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "original_doc_id": doc_id,
                            "content_hash": hashlib.md5(chunk.encode('utf-8')).hexdigest()[:16]
                        },
                        "type": doc_type
                    })
        
        logger.info(f"文档分块完成，共 {len(chunked_docs)} 个块，发现 {duplicate_counter} 个潜在重复")
        
        # 检查是否有重复ID
        all_ids = [doc["id"] for doc in chunked_docs]
        unique_ids = set(all_ids)
        if len(all_ids) != len(unique_ids):
            logger.warning(f"发现重复ID: {len(all_ids) - len(unique_ids)} 个重复")
            # 找到重复的ID
            from collections import Counter
            duplicates = [item for item, count in Counter(all_ids).items() if count > 1]
            logger.warning(f"重复ID示例: {duplicates[:10]}")
            
            # 特别检查特定的重复ID
            if "6663d9a70581b72e" in duplicates:
                logger.error(f"找到特定重复ID 6663d9a70581b72e 的详细信息:")
                for doc in chunked_docs:
                    if doc["id"] == "6663d9a70581b72e":
                        logger.error(f"文档内容前100字符: {doc['content'][:100]}")
                        logger.error(f"文档元数据: {doc['metadata']}")
            
            # 修复重复ID
            chunked_docs = self.fix_duplicate_ids(chunked_docs)
        
        self.stats["total_chunks"] = len(chunked_docs)
        
        return chunked_docs
    
    def fix_duplicate_ids(self, chunked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """修复重复ID"""
        id_count = {}
        fixed_docs = []
        
        for doc in chunked_docs:
            doc_id = doc["id"]
            if doc_id not in id_count:
                id_count[doc_id] = 0
                fixed_docs.append(doc)
            else:
                id_count[doc_id] += 1
                # 为重复ID添加后缀
                new_id = f"{doc_id}_dup{id_count[doc_id]}_{uuid.uuid4().hex[:6]}"
                fixed_doc = doc.copy()
                fixed_doc["id"] = new_id
                fixed_docs.append(fixed_doc)
                logger.info(f"修复重复ID: {doc_id} -> {new_id}")
        
        logger.info(f"已修复 {sum(id_count.values())} 个重复ID")
        return fixed_docs
    
    @staticmethod
    def split_text(text: str, max_length: int) -> List[str]:
        """按句子分割文本"""
        import re
        
        # 按句子分割（考虑中文标点）
        sentences = re.split(r'[。！？；;.!?]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 如果当前块加上新句子会超过限制，就保存当前块
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # 单个句子就超过限制，强制分割
                    chunks.append(sentence[:max_length])
                    current_chunk = sentence[max_length:] if len(sentence) > max_length else ""
            else:
                if current_chunk:
                    current_chunk += "。" + sentence if sentence[-1] not in "。！？；;.!?" else sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def build_vector_database(self, batch_size: int = 100):
        """构建向量数据库"""
        logger.info("开始构建向量数据库...")
        
        # 初始化模型和数据库
        self.initialize_embedding_model()
        self.initialize_chroma_db()
        
        # 文档分块
        chunked_docs = self.chunk_documents()
        
        if not chunked_docs:
            logger.error("没有文档需要处理")
            return
        
        # 分批处理文档
        total_chunks = len(chunked_docs)
        logger.info(f"开始向量化处理，共 {total_chunks} 个文档块")
        
        # 重置ID计数器
        self.id_counter = {}
        
        # 先收集所有ID，检查是否有重复
        all_ids = [doc["id"] for doc in chunked_docs]
        if len(all_ids) != len(set(all_ids)):
            logger.error(f"仍然存在重复ID: {len(all_ids) - len(set(all_ids))} 个重复")
            # 找到重复的ID
            from collections import Counter
            duplicates = [item for item, count in Counter(all_ids).items() if count > 1]
            logger.error(f"重复ID: {duplicates[:20]}")
            
            # 使用新的UUID替换所有重复ID
            id_map = {}
            for i, doc_id in enumerate(all_ids):
                if doc_id in duplicates:
                    # 对于重复的ID，使用UUID替换
                    new_id = f"{doc_id}_fixed_{uuid.uuid4().hex[:8]}"
                    chunked_docs[i]["id"] = new_id
                    id_map[doc_id] = new_id
                    logger.info(f"替换重复ID: {doc_id} -> {new_id}")
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="向量化处理", unit="batch"):
            batch = chunked_docs[i:i+batch_size]
            
            # 提取内容和元数据
            contents = [doc["content"] for doc in batch]
            ids = [doc["id"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            # 检查批次内是否有重复ID
            if len(ids) != len(set(ids)):
                logger.warning(f"批次 {i//batch_size} 发现重复ID，正在修复...")
                # 修复批次内重复ID
                id_map = {}
                for j, doc_id in enumerate(ids):
                    if doc_id not in id_map:
                        id_map[doc_id] = 0
                    else:
                        id_map[doc_id] += 1
                        ids[j] = f"{doc_id}_batch_dup_{id_map[doc_id]}_{uuid.uuid4().hex[:6]}"
                        metadatas[j]["fixed_id"] = True
            
            # 生成向量
            embeddings = self.embedding_model.encode(
                contents,
                batch_size=min(batch_size, 32),
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # 添加到集合
            try:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.error(f"添加批次 {i//batch_size} 失败: {e}")
                
                # 记录详细的错误信息
                if "6663d9a70581b72e" in str(e):
                    logger.error(f"特定错误：ID 6663d9a70581b72e 重复")
                    # 找出这个ID在哪个位置
                    for j, doc_id in enumerate(ids):
                        if doc_id == "6663d9a70581b72e":
                            logger.error(f"重复ID位置: 批次 {i//batch_size}, 索引 {j}")
                            logger.error(f"内容前100字符: {contents[j][:100]}")
                
                # 尝试逐个添加
                successful_adds = 0
                for j in range(len(batch)):
                    try:
                        self.collection.add(
                            embeddings=[embeddings[j].tolist()],
                            documents=[contents[j]],
                            metadatas=[metadatas[j]],
                            ids=[ids[j]]
                        )
                        successful_adds += 1
                    except Exception as e2:
                        logger.warning(f"添加单个文档失败 {ids[j]}: {e2}")
                        # 尝试使用新的UUID
                        try:
                            new_id = f"uuid_{uuid.uuid4().hex}"
                            self.collection.add(
                                embeddings=[embeddings[j].tolist()],
                                documents=[contents[j]],
                                metadatas=[metadatas[j]],
                                ids=[new_id]
                            )
                            successful_adds += 1
                            logger.info(f"使用新UUID {new_id} 添加成功")
                        except Exception as e3:
                            logger.error(f"重试也失败: {e3}")
                
                logger.info(f"批次 {i//batch_size} 成功添加 {successful_adds}/{len(batch)} 个文档")
        
        # 保存统计信息
        self.save_statistics(chunked_docs)
        
        logger.info(f"✅ 向量数据库构建完成！")
        logger.info(f"   共存储 {total_chunks} 个文档块")
        logger.info(f"   存储位置: {self.chroma_persist_dir}")
        
        # 测试检索
        self.test_retrieval()
    
    def save_statistics(self, chunked_docs: List[Dict[str, Any]]):
        """保存构建统计信息"""
        stats = {
            "total_documents": len(self.documents),
            "total_chunks": len(chunked_docs),
            "embedding_model": "BAAI/bge-m3",
            "embedding_dimension": self.stats["embedding_dimension"],
            "llm_model": "Qwen2.5-3B-Instruct",
            "chroma_persist_dir": self.chroma_persist_dir,
            "device": self.device,
            "build_time": self.get_current_time(),
            "environment": "阿里云PAI DSW"
        }
        
        # 按文档类型统计
        type_stats = {}
        for doc in chunked_docs:
            doc_type = doc.get("type", "unknown")
            type_stats[doc_type] = type_stats.get(doc_type, 0) + 1
        
        stats["documents_by_type"] = type_stats
        
        # 保存到文件（放在OSS中供后续使用）
        stats_file = os.path.join(os.path.dirname(self.knowledge_base_path), "vector_db_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存到: {stats_file}")
        
        # 同时在本地保存一份
        local_stats_file = os.path.join(self.chroma_persist_dir, "build_statistics.json")
        with open(local_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    
    def get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def test_retrieval(self, n_queries: int = 3):
        """测试检索功能"""
        if not self.collection:
            logger.error("向量数据库未初始化")
            return
        
        test_queries = [
            "公司的法定代表人是谁？",
            "产品的价格是多少？",
            "有哪些招标项目？"
        ][:n_queries]
        
        logger.info("开始检索测试...")
        
        for query in test_queries:
            # 将查询转换为向量
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True
            )
            
            # 检索相似文档
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                logger.info(f"\n查询: '{query}'")
                logger.info(f"检索到 {len(results['documents'][0])} 个相关文档:")
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    logger.info(f"  文档 {i+1} (相似度: {1-distance:.4f}): {doc[:80]}...")
                    
            except Exception as e:
                logger.error(f"查询 '{query}' 失败: {e}")
    
    def backup_to_oss(self, oss_backup_dir: str = None):
        """将向量数据库备份到OSS（可选）"""
        if not oss_backup_dir:
            # 默认备份到知识库同级目录
            oss_backup_dir = os.path.join(os.path.dirname(self.knowledge_base_path), "chroma_db_backup")
        
        logger.info(f"备份向量数据库到OSS: {oss_backup_dir}")
        
        try:
            # 创建OSS备份目录
            os.makedirs(oss_backup_dir, exist_ok=True)
            
            # 复制ChromaDB目录到OSS
            if os.path.exists(self.chroma_persist_dir):
                backup_path = os.path.join(oss_backup_dir, os.path.basename(self.chroma_persist_dir))
                
                # 使用shutil复制目录
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                
                shutil.copytree(self.chroma_persist_dir, backup_path)
                logger.info(f"✅ 备份完成: {backup_path}")
            else:
                logger.error(f"源目录不存在: {self.chroma_persist_dir}")
                
        except Exception as e:
            logger.error(f"备份失败: {e}")
    
    def analyze_knowledge_base(self):
        """分析知识库，找出可能的问题"""
        logger.info("分析知识库内容...")
        
        # 检查是否有重复内容
        content_hashes = {}
        duplicate_contents = []
        
        for doc_idx, doc in enumerate(self.documents[:1000]):  # 只检查前1000个文档
            content = doc.get("content", "")
            if not content:
                continue
            
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in content_hashes:
                duplicate_contents.append({
                    "doc_idx": doc_idx,
                    "content": content[:200],
                    "previous_idx": content_hashes[content_hash]
                })
            else:
                content_hashes[content_hash] = doc_idx
        
        if duplicate_contents:
            logger.warning(f"发现 {len(duplicate_contents)} 个重复内容")
            for dup in duplicate_contents[:5]:
                logger.warning(f"文档 {dup['doc_idx']} 与文档 {dup['previous_idx']} 内容相同")
                logger.warning(f"内容预览: {dup['content']}...")
        
        # 检查文档长度
        doc_lengths = [len(doc.get("content", "")) for doc in self.documents]
        avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        logger.info(f"平均文档长度: {avg_length:.2f} 字符")
        
        # 检查是否有空的文档ID
        empty_doc_ids = sum(1 for doc in self.documents if not doc.get("doc_id"))
        if empty_doc_ids:
            logger.warning(f"有 {empty_doc_ids} 个文档没有doc_id")
    
    def build(self):
        """完整构建流程"""
        logger.info("=" * 60)
        logger.info("开始DSW环境向量数据库构建流程")
        logger.info("=" * 60)
        
        try:
            # 分析知识库
            self.analyze_knowledge_base()
            
            # 构建向量数据库
            self.build_vector_database()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ 向量数据库构建成功！")
            logger.info("=" * 60)
            
            # 显示使用说明
            logger.info("\n使用说明:")
            logger.info(f"1. ChromaDB目录: {self.chroma_persist_dir}")
            logger.info(f"2. 知识库文档数: {len(self.documents)}")
            logger.info(f"3. 向量块数: {self.stats['total_chunks']}")
            logger.info(f"4. 嵌入维度: {self.stats['embedding_dimension']}")
            
            # 询问是否备份到OSS
            logger.info("\n是否备份向量数据库到OSS? (y/n)")
            try:
                import sys
                response = sys.stdin.readline().strip().lower()
                if response == 'y':
                    self.backup_to_oss()
            except:
                pass
            
        except Exception as e:
            logger.error(f"构建过程中出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DSW环境RAG向量数据库构建")
    
    parser.add_argument('--knowledge-base', type=str, required=True,
                       help='知识库JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='/tmp/chroma_db_dsw',
                       help='ChromaDB输出目录（使用容器本地存储）')
    parser.add_argument('--embedding-path', type=str, 
                       default='/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3',
                       help='Embedding模型本地路径')
    parser.add_argument('--llm-path', type=str, 
                       default='/mnt/workspace/data/modelscope/cache/qwen/Qwen2___5-3B-Instruct',
                       help='大语言模型本地路径')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='运行设备，默认自动检测')
    
    args = parser.parse_args()
    
    # 创建构建器
    builder = DSWVectorDatabaseBuilder(
        knowledge_base_path=args.knowledge_base,
        embedding_local_path=args.embedding_path,
        llm_local_path=args.llm_path,
        chroma_persist_dir=args.output_dir,
        device=args.device
    )
    
    # 构建向量数据库
    builder.build()


if __name__ == "__main__":
    main()