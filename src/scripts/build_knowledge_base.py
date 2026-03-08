#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于真实数据库记录构建高效RAG知识库
"""

import os
import pandas as pd
import json
from sqlalchemy import create_engine, text
import argparse
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
import hashlib
from datetime import datetime
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_knowledge_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KnowledgeDocumentBuilder:
    """根据不同类型的数据构建知识文档
"""清理文本数据，移除多余空格和不可见字符，并移除所有空格"""if pd.isna(text):
            return ""
        
        text = str(text).strip()
        # 将各种空白字符（换行、制表等）替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除控制字符
        text = ''.join(char for char in text if char.isprintable())
        # 移除所有空格（包括中英文间的空格），解决生成错别字问题
        text = text.replace(' ', '')
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int) -> str:
        """截断文本并添加省略号"""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def generate_doc_id(content: str, source: str = "") -> str:
        """生成文档唯一ID"""
        doc_str = f"{source}:{content}"
        return hashlib.md5(doc_str.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def assign_weight(doc_type: str) -> float:
        """为文档分配检索权重"""
        return KnowledgeDocumentBuilder.DOCUMENT_WEIGHTS.get(doc_type, 0.5)
    
    @staticmethod
    def smart_select_info_parts(info_dict: Dict[str, str], max_parts: int = None) -> List[str]:
        """智能选择最重要的信息字段"""
        if max_parts is None:
            max_parts = KnowledgeDocumentBuilder.MAX_INFO_FIELDS
        
        # 定义字段重要性排序
        field_priority = {
            'company': ['company_name', 'legal_representative', 'province', 'city', 
                       'registration_address', 'credit_code', 'registered_capital_value'],
            'product': ['product_name', 'price', 'supplier', 'product_category', 
                       'keyword', 'address', 'brand'],
            'price': ['product_name', 'price', 'supplier', 'product_category', 'keyword'],
            'zhaobiao': ['project_name', 'buyer', 'supplier', 'contract_amount', 
                        'contract_name', 'procurement_method'],
            'zhongbiao': ['project_name', 'buyer', 'supplier', 'contract_amount', 
                         'contract_name', 'procurement_method'],
            'law': ['title', 'law_name', 'law_type', 'publish_date', 'law_code']
        }
        
        # 找出info_dict属于哪个类型
        info_type = None
        for t, fields in field_priority.items():
            if any(field in info_dict for field in fields):
                info_type = t
                break
        
        if not info_type:
            # 如果没有匹配类型，按原顺序取前max_parts个
            info_items = list(info_dict.items())[:max_parts]
            return [f"{k}: {v}" for k, v in info_items]
        
        # 按优先级排序
        prioritized_items = []
        for field in field_priority[info_type]:
            if field in info_dict and info_dict[field]:
                prioritized_items.append((field, info_dict[field]))
        
        # 补充剩余字段
        remaining_fields = [(k, v) for k, v in info_dict.items() 
                          if k not in [item[0] for item in prioritized_items] and v]
        prioritized_items.extend(remaining_fields[:max_parts
"""构建公司相关文档 - 优化版"""documents = []
        
        # 提取所有company表字段
        company_id = KnowledgeDocumentBuilder.clean_text(row.get('id', ''))
        company_name = KnowledgeDocumentBuilder.clean_text(row.get('company_name', ''))
        credit_code = KnowledgeDocumentBuilder.clean_text(row.get('credit_code', ''))
        registration_date = KnowledgeDocumentBuilder.clean_text(row.get('registration_date', ''))
        company_type = KnowledgeDocumentBuilder.clean_text(row.get('company_type', ''))
        legal_rep = KnowledgeDocumentBuilder.clean_text(row.get('legal_representative', ''))
        legal_rep_pinyin = KnowledgeDocumentBuilder.clean_text(row.get('legal_representative_pinyin', ''))
        registered_capital = KnowledgeDocumentBuilder.clean_text(row.get('registered_capital_value', ''))
        province = KnowledgeDocumentBuilder.clean_text(row.get('province', ''))
        city = KnowledgeDocumentBuilder.clean_text(row.get('city', ''))
        registration_address = KnowledgeDocumentBuilder.clean_text(row.get('registration_address', ''))
        business_scope = KnowledgeDocumentBuilder.truncate_text(
            KnowledgeDocumentBuilder.clean_text(row.get('business_scope', '')),
            KnowledgeDocumentBuilder.MAX_SCOPE_LENGTH
        )
        
        # 数据验证
        if not company_name or company_name == "未知" or company_name == "未填写":
            return documents
        
        # 文档1: 公司完整信息（智能选择字段）
        info_dict = {
            "公司名称": company_name,
            "法定代表人": legal_rep,
            "统一社会信用代码": credit_code,
            "注册地址": registration_address if registration_address else f"{province}{city}",
            "注册资本": registered_capital,
            "公司类型": company_type,
            "经营范围": business_scope,
            "注册日期": registration_date
        }
        
        # 过滤空值
        info_dict = {k: v for k, v in info_dict.items() if v}
        
        # 智能选择字段
        info_parts = KnowledgeDocumentBuilder.smart_select_info_parts(info_dict)
        
        if info_parts:
            content = f"{company_name}，{'，'.join(info_parts)}"
            
            # 截断过长的内容
            if len(content) > KnowledgeDocumentBuilder.MAX_CONTENT_LENGTH:
                content = content[:KnowledgeDocumentBuilder.MAX_CONTENT_LENGTH-3] + "..."
            
            documents.append({
                "content": content,
                "type": "company_full",
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, "company"),
                "weight": KnowledgeDocumentBuilder.assign_weight("company_full"),
                "metadata": {
                    "company_id": company_id,
                    "company_name": company_name,
                    "credit_code": credit_code,
                    "legal_representative": legal_rep,
                    "province": province,
                    "city": city,
                    "registration_address": registration_address,
                    "company_type": company_type,
                    "business_scope": business_scope
                }
            })
        
        # 文档2: 法人查询（仅当有法人信息时）
        if legal_rep and legal_rep != "无" and legal_rep != "未填写":
            content = f"{company_name}的法定代表人是{legal_rep}"
            documents.append({
                "content": content,
                "type": "company_legal_rep",
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, "company"),
                "weight": KnowledgeDocumentBuilder.assign_weight("company_legal_rep"),
                "metadata": {
                    "company_name": company_name,
                    "legal_representative": legal_rep
                }
            })
        
        # 文档3: 公司地点（多种表达方式）
        location_variants = []
        
        if registration_address:
            location_variants.append(f"{company_name}的注册地址是{registration_address}")
        
        if province and city:
            location_variants.append(f"{company_name}位于{province}{city}")
        elif province:
            location_variants.append(f"{company_name}位于{province}")
        
        for variant in location_variants[:2]:  # 最多生成2个地点变体
            documents.append({
                "content": variant,
                "type": "company_location",
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(variant, "company"),
                "weight": KnowledgeDocumentBuilder.assign_weight("company_location"),
                "metadata": {
                    "company_name": company_name,
                    "province": province,
                    "city": city,
                    "registration_address": registration_address
                }
            })
        
        # 文档4: 公司信用代码（仅当有信用代码时）
        if credit_code and len(credit_code) >= 15:  # 基本验证
            content = f"{company_name}的统一社会信用代码是{credit_code}"
            documents.append({
                "content": content,
                "type": "company_credit_code",
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, "company"),
                "weight": KnowledgeDocumentBuilder.assign_weight("company_credit_code"),
                "metadata": {
                    "company_name": company_name,
                    "credit_code": credit_code
                }
            })
        
        return documents
    
    @staticmethod
    def build_product_documents(row: pd.Series) -> List[Dict[str, Any]]:
        """构建产品相关文档
"""构建价格相关文档 - 优化版"""documents = []
        
        # 提取字段
        product_name = KnowledgeDocumentBuilder.clean_text(row.get('product_name', ''))
        price = KnowledgeDocumentBuilder.clean_text(row.get('price', ''))
        supplier = KnowledgeDocumentBuilder.clean_text(row.get('supplier', ''))
        source = KnowledgeDocumentBuilder.clean_text(row.get('source', ''))
        product_category = KnowledgeDocumentBuilder.clean_text(row.get('product_category', ''))
        keyword = KnowledgeDocumentBuilder.clean_text(row.get('keyword', ''))
        
        # 数据验证
        if not product_name or not price:
            return documents
        
        # 尝试转换为数字验证价格有效性
        try:
            price_num = float(str(price).replace(',', '').replace('元', '').replace('￥', ''))
            if price_num <= 0 or price_num > 100000000:  # 价格范围验证
                logger.debug(f"价格异常: {product_name}
"""构建招标相关文档 - 优化版"""return KnowledgeDocumentBuilder._build_bid_documents(row, "zhaobiao")
    
    @staticmethod
    def build_zhongbiao_documents(row: pd.Series) -> List[Dict[str, Any]]:
        """构建中标相关文档
"""构建招标/中标文档通用方法"""documents = []
        
        # 提取核心字段
        project_name = KnowledgeDocumentBuilder.clean_text(row.get('project_name', ''))
        contract_amount = KnowledgeDocumentBuilder.clean_text(row.get('contract_amount', ''))
        buyer = KnowledgeDocumentBuilder.clean_text(row.get('buyer', ''))
        supplier = KnowledgeDocumentBuilder.clean_text(row.get('supplier', ''))
        contract_name = KnowledgeDocumentBuilder.clean_text(row.get('contract_name', ''))
        
        # 数据验证
        if not project_name:
            return documents
        
        # 文档1: 项目核心信息
        info_parts = []
        if contract_name and contract_name != project_name:
            info_parts.append(f"合同名称: {contract_name}")
        
        if buyer:
            info_parts.append(f"采购方: {buyer}")
        
        if supplier:
            prefix = "中标供应商: " if bid_type == "zhongbiao" else "供应商: "
            info_parts.append(f"{prefix}{supplier}")
        
        if contract_amount:
            info_parts.append(f"合同金额: {contract_amount}元")
        
        if info_parts:
            project_type = "中标项目" if bid_type == "zhongbiao" else "招标项目"
            content = f"{project_type}{project_name}，{'，'.join(info_parts[:4])}"  # 最多4个信息
            
            # 截断过长的内容
            if len(content) > KnowledgeDocumentBuilder.MAX_CONTENT_LENGTH:
                content = content[:KnowledgeDocumentBuilder.MAX_CONTENT_LENGTH-3] + "..."
            
            doc_type = f"{bid_type}_full"
            documents.append({
                "content": content,
                "type": doc_type,
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, bid_type),
                "weight": KnowledgeDocumentBuilder.assign_weight(doc_type),
                "metadata": {
                    "project_name": project_name,
                    "buyer": buyer,
                    "supplier": supplier,
                    "contract_amount": contract_amount,
                    "contract_name": contract_name
                }
            })
        
        # 文档2: 供应商信息
        if supplier:
            prefix = "中标供应商是" if bid_type == "zhongbiao" else "供应商是"
            content = f"{project_name}的{prefix}{supplier}"
            
            doc_type = f"{bid_type}_supplier"
            documents.append({
                "content": content,
                "type": doc_type,
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, bid_type),
                "weight": KnowledgeDocumentBuilder.assign_weight(doc_type),
                "metadata": {
                    "project_name": project_name,
                    "supplier": supplier
                }
            })
        
        # 文档3: 采购方信息
        if buyer:
            content = f"{project_name}的采购方是{buyer}"
            
            doc_type = f"{bid_type}_buyer"
            documents.append({
                "content": content,
                "type": doc_type,
                "doc_id": KnowledgeDocumentBuilder.generate_doc_id(content, bid_type),
                "weight": KnowledgeDocumentBuilder.assign_weight(doc_type),
                "metadata": {
                    "project_name": project_name,
                    "buyer": buyer
                }
            })
        
        return documents
    
    @staticmethod
    def build_law_documents(row: pd.Series) -> List[Dict[str, Any]]:
        """构建法规相关文档
"""知识库构建主程序 - 优化版"""def __init__(self, db_config: Dict[str, Any], batch_size: int = 5000, 
                 output_dir: str = "rag_knowledge_base", 
                 save_batch_size: int = 10000):
        """初始化
        
        Args:
            _config: 
            batch_size: 
            output_dir: 
            save_batch_size: 
        """
        self.db_config = db_config
        self.batch_size = batch_size
        self.save_batch_size = save_batch_size
        
        # 添加时间戳到输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}_{timestamp}"
        
        self.engine = self._create_db_engine()
        self.document_builder = KnowledgeDocumentBuilder()
        
        # 统计数据
        self.stats = {
            "total_documents": 0,
            "tables_processed": 0,
            "documents_by_table": {},
            "documents_by_type": {},
            "invalid_records": 0,
            "duplicate_documents": 0
        }
        
        # 表字段映射（只选择关键字段，提高性能）
        self.table_essential_columns = {
            "company": [
                'id', 'company_name', 'credit_code', 'registration_date',
                'company_type', 'legal_representative', 'registered_capital_value',
                'province', 'city', 'registration_address', 'business_scope'
            ],
            "product": [
                'id', 'product_name', 'keyword', 'price', 'supplier', 
                'address', 'description', 'product_category', 'specification', 
                'brand', 'supplier_2'
            ],
            "price": [
                'id', 'product_name', 'price', 'supplier', 'source',
                'product_category', 'keyword', 'collection_time'
            ],
            "zhaobiao": [
                'id', 'project_name', 'contract_name', 'contract_amount',
                'buyer', 'supplier', 'procurement_method'
            ],
            "zhongbiao": [
                'id', 'project_name', 'contract_name', 'contract_amount',
                'buyer', 'supplier', 'procurement_method'
            ],
            "law": [
                'id', 'title', 'law_name', 'law_type', 'content',
                'publish_date', 'law_code'
            ]
        }
        
        # 表处理映射
        self.table_handlers = {
            "company": self.process_company_table,
            "product": self.process_product_table,
            "price": self.process_price_table,
            "law": self.process_law_table,
            "zhaobiao": self.process_zhaobiao_table,
            "zhongbiao": self.process_zhongbiao_table
        }
        
        # 创建输出目录结构
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "batches"), exist_ok=True)
        
        logger.info(f"输出目录已创建: {self.output_dir}")
        
        # 用于文档去重
        self.seen_documents: Set[str] = set()
    
    def _create_db_engine(self):
        """创建数据库引擎"""
        try:
            engine = create_engine(
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/"
                f"{self.db_config['database']}?charset={self.db_config['charset']}",
                pool_size=10,  # 连接池大小
                max_overflow=20,  # 最大溢出连接数
                pool_recycle=3600  # 连接回收时间（秒）
            )
            logger.info("数据库引擎创建成功")
            return engine
        except Exception as e:
            logger.error(f"数据库引擎创建失败: {e}")
            raise
    
    def get_table_row_count(self, table_name: str) -> int:
        """获取表记录数"""
        try:
            with self.engine.connect() as conn:
                query = text(f"SELECT COUNT(*) as count FROM {table_name}")
                result = conn.execute(query).fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"获取表 {table_name} 记录数失败: {e}")
            return 0
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """获取表字段列表"""
        try:
            with self.engine.connect() as conn:
                # 获取表结构信息
                if self.db_config['database']:
                    query = text(f"""
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = '{self.db_config['database']}' 
                        AND TABLE_NAME = '{table_name}'
                    """)
                else:
                    query = text(f"SHOW COLUMNS FROM {table_name}")
                
                result = conn.execute(query).fetchall()
                return [row[0] for row in result] if result else []
        except Exception as e:
            logger.warning(f"获取表 {table_name} 字段列表失败: {e}")
            return []
    
    def validate_row_data(self, table_name: str, row: pd.Series) -> bool:
        """验证行数据有效性"""
        try:
            # 不同表的不同验证规则
            if table_name == "company":
                required_fields = ['company_name']
                return all(pd.notna(row.get(field, '')) and str(row[field]).strip() 
                          for field in required_fields)
            
            elif table_name == "product":
                required_fields = ['product_name']
                return all(pd.notna(row.get(field, '')) and str(row[field]).strip() 
                          for field in required_fields)
            
            elif table_name == "price":
                required_fields = ['product_name', 'price']
                return all(pd.notna(row.get(field, '')) and str(row[field]).strip() 
                          for field in required_fields)
            
            # 其他表有更宽松的验证
            return True
            
        except Exception as e:
            logger.debug(f"数据验证失败: {e}")
            return False
    
    def deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于内容去重文档"""
        unique_docs = []
        duplicates_removed = 0
        
        for doc in documents:
            # 使用内容和类型组合作为去重键
            dedup_key = f"{doc['type']}:{doc['content']}"
            
            if dedup_key not in self.seen_documents:
                self.seen_documents.add(dedup_key)
                unique_docs.append(doc)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.debug(f"移除 {duplicates_removed} 个重复文档")
            self.stats["duplicate_documents"] += duplicates_removed
        
        return unique_docs
    
    def read_table_data(self, table_name: str, columns: List[str] = None, 
                       limit: int = None) -> pd.DataFrame:
        """读取表数据（优化字段选择）"""
        try:
            # 如果没有指定字段，使用预定义的关键字段
            if columns is None:
                columns = self.table_essential_columns.get(table_name)
                if not columns:
                    # 如果预定义字段不存在，获取所有字段
                    columns = self.get_table_columns(table_name)
            
            if not columns:
                logger.warning(f"表 {table_name} 没有可用字段")
                return pd.DataFrame()
            
            with self.engine.connect() as conn:
                cols_str = ', '.join(columns)
                query = f"SELECT {cols_str} FROM {table_name}"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql(text(query), conn)
                logger.info(f"读取表 {table_name} 成功，共 {len(df)} 条记录，{len(columns)} 个字段")
                return df
                
        except Exception as e:
            logger.error(f"读取表 {table_name} 失败: {e}")
            # 尝试使用SELECT *作为备选方案
            try:
                with self.engine.connect() as conn:
                    query = f"SELECT * FROM {table_name}"
                    if limit:
                        query += f" LIMIT {limit}"
                    df = pd.read_sql(text(query), conn)
                    logger.warning(f"使用SELECT *读取表 {table_name} 成功，共 {len(df)} 条记录")
                    return df
            except Exception as e2:
                logger.error(f"备选读取方案也失败: {e2}")
                return pd.DataFrame()
    
    def load_table_data_batch(self, table_name: str) -> pd.DataFrame:
        """分批加载表数据，避免内存溢出"""
        all_data = []
        offset = 0
        
        # 先获取总记录数
        total_rows = self.get_table_row_count(table_name)
        logger.info(f"表 {table_name} 总记录数: {total_rows:,}")
        
        if total_rows == 0:
            logger.warning(f"表 {table_name} 没有数据")
            return pd.DataFrame()
        
        # 如果记录数小于等于批次大小，直接一次性读取
        if total_rows <= self.batch_size:
            logger.info(f"表 {table_name} 记录数较少，直接一次性读取")
            return self.read_table_data(table_name)
        
        # 获取关键字段
        essential_columns = self.table_essential_columns.get(table_name)
        
        # 分批读取
        pbar = tqdm(total=total_rows, desc=f"加载 {table_name}")
        
        while offset < total_rows:
            try:
                with self.engine.connect() as conn:
                    # 构建查询
                    if essential_columns:
                        cols_str = ', '.join(essential_columns)
                        base_query = f"SELECT {cols_str} FROM {table_name}"
                    else:
                        base_query = f"SELECT * FROM {table_name}"
                    
                    query = text(f"{base_query} LIMIT {self.batch_size} OFFSET {offset}")
                    batch_df = pd.read_sql(query, conn)
                    
                    if not batch_df.empty:
                        all_data.append(batch_df)
                        loaded_count = min(offset + self.batch_size, total_rows)
                        pbar.update(len(batch_df))
                        pbar.set_postfix({"进度": f"{loaded_count:,}/{total_rows:,}"})
                    
                    offset += self.batch_size
                    
            except Exception as e:
                logger.error(f"加载批次失败 (offset={offset}): {e}")
                # 尝试跳过这个批次
                offset += self.batch_size
        
        pbar.close()
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"表 {table_name} 数据加载完成，共 {len(combined_df):,} 条记录")
            return combined_df
        else:
            logger.warning(f"表 {table_name} 没有成功加载数据")
            return pd.DataFrame()
    
    def save_documents_batch(self, documents: List[Dict[str, Any]], 
                           batch_num: int, table_name: str = None):
        """分批保存文档到文件"""
        if not documents:
            return
        
        # 创建批次文件名
        if table_name:
            batch_file = f"{self.output_dir}/batches/{table_name}_batch_{batch_num:04d}.json"
        else:
            batch_file = f"{self.output_dir}/batches/batch_{batch_num:04d}.json"
        
        # 保存批次数据
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump({
                "batch_num": batch_num,
                "table_name": table_name,
                "document_count": len(documents),
                "created_at": datetime.now().isoformat(),
                "documents": documents
            }, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"批次 {batch_num} 已保存: {batch_file} ({len(documents)} 个文档)")
    
    def process_table_with_batch_saving(self, table_name: str, 
                                      process_func) -> List[Dict[str, Any]]:
        """处理表数据并分批保存"""
        logger.info(f"开始处理表: {table_name}")
        df = self.load_table_data_batch(table_name)
        
        if df.empty:
            logger.warning(f"表 {table_name} 为空")
            return []
        
        # 保存原始数据（用于调试）
        raw_file = f"{self.output_dir}/raw/{table_name}_raw_sample.csv"
        df.head(1000).to_csv(raw_file, index=False, encoding='utf-8')
        logger.info(f"原始数据样本已保存: {raw_file}")
        
        all_documents = []
        batch_documents = []
        batch_num = 1
        valid_records = 0
        invalid_records = 0
        
        # 处理每一行数据
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {table_name}"):
            try:
                # 数据验证
                if not self.validate_row_data(table_name, row):
                    invalid_records += 1
                    continue
                
                # 生成文档
                docs = process_func(row)
                
                if docs:
                    # 去重
                    unique_docs = self.deduplicate_documents(docs)
                    
                    if unique_docs:
                        batch_documents.extend(unique_docs)
                        valid_records += 1
                        
                        # 达到保存批次大小时保存
                        if len(batch_documents) >= self.save_batch_size:
                            self.save_documents_batch(batch_documents, batch_num, table_name)
                            all_documents.extend(batch_documents)
                            batch_documents = []
                            batch_num += 1
                
            except Exception as e:
                logger.error(f"处理行 {idx} 时出错: {e}")
                invalid_records += 1
        
        # 保存剩余的文档
        if batch_documents:
            self.save_documents_batch(batch_documents, batch_num, table_name)
            all_documents.extend(batch_documents)
        
        # 更新统计信息
        self.stats["documents_by_table"][table_name] = len(all_documents)
        self.stats["invalid_records"] += invalid_records
        
        for doc in all_documents:
            doc_type = doc.get('type', 'unknown')
            self.stats["documents_by_type"][doc_type] = self.stats["documents_by_type"].get(doc_type, 0) + 1
        
        logger.info(f"表 {table_name} 处理完成: "
                   f"{len(all_documents)} 个文档, "
                   f"{valid_records} 有效记录, "
                   f"{invalid_records} 无效记录")
        
        return all_documents
    
    def process_company_table(self, table_name: str = "company") -> List[Dict[str, Any]]:
        """处理公司表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_company_documents
        )
    
    def process_product_table(self, table_name: str = "product") -> List[Dict[str, Any]]:
        """处理产品表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_product_documents
        )
    
    def process_price_table(self, table_name: str = "price") -> List[Dict[str, Any]]:
        """处理价格表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_price_documents
        )
    
    def process_law_table(self, table_name: str = "law") -> List[Dict[str, Any]]:
        """处理法规表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_law_documents
        )
    
    def process_zhaobiao_table(self, table_name: str = "zhaobiao") -> List[Dict[str, Any]]:
        """处理招标表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_zhaobiao_documents
        )
    
    def process_zhongbiao_table(self, table_name: str = "zhongbiao") -> List[Dict[str, Any]]:
        """处理中标表"""
        return self.process_table_with_batch_saving(
            table_name, 
            self.document_builder.build_zhongbiao_documents
        )
    
    def merge_batch_files(self, tables: List[str] = None) -> List[Dict[str, Any]]:
        """合并所有批次文件"""
        all_documents = []
        batches_dir = f"{self.output_dir}/batches"
        
        if not os.path.exists(batches_dir):
            logger.warning(f"批次目录不存在: {batches_dir}")
            return all_documents
        
        # 获取所有批次文件
        batch_files = sorted([f for f in os.listdir(batches_dir) if f.endswith('.json')])
        
        if not batch_files:
            logger.warning("没有找到批次文件")
            return all_documents
        
        logger.info(f"开始合并 {len(batch_files)} 个批次文件")
        
        for batch_file in tqdm(batch_files, desc="合并批次"):
            try:
                file_path = os.path.join(batches_dir, batch_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                if "documents" in batch_data:
                    all_documents.extend(batch_data["documents"])
                
            except Exception as e:
                logger.error(f"读取批次文件失败 {batch_file}: {e}")
        
        logger.info(f"批次合并完成，共 {len(all_documents)} 个文档")
        return all_documents
    
    def build_knowledge_base(self, tables: List[str] = None) -> Dict[str, Any]:
        """构建完整知识库"""
        if tables is None:
            tables = list(self.table_handlers.keys())
        
        logger.info(f"开始构建知识库，处理表: {tables}")
        
        # 处理每个表
        for table in tables:
            if table in self.table_handlers:
                try:
                    logger.info(f"处理表: {table}")
                    documents = self.table_handlers[table]()
                    self.stats["tables_processed"] += 1
                    logger.info(f"表 {table} 完成，生成 {len(documents)} 个文档")
                    
                except Exception as e:
                    logger.error(f"处理表 {table} 时出错: {e}")
                    continue
            else:
                logger.warning(f"表 {table} 没有对应的处理函数")
        
        # 合并所有批次文件
        all_documents = self.merge_batch_files(tables)
        self.stats["total_documents"] = len(all_documents)
        
        # 保存最终知识库
        self.save_knowledge_base(all_documents)
        
        # 清理批次文件（可选）
        self.cleanup_batch_files()
        
        return {
            "documents": all_documents,
            "stats": self.stats
        }
    
    def save_knowledge_base(self, documents: List[Dict[str, Any]]):
        """保存知识库到文件"""
        
        # 1. 保存为文本文件（每行一个文档）
        txt_file = f"{self.output_dir}/knowledge_base.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(doc['content'] + '\n')
        logger.info(f"知识库文本文件已保存: {txt_file} ({len(documents):,} 行)")
        
        # 2. 保存为JSON文件（包含元数据和权重）
        json_file = f"{self.output_dir}/knowledge_base.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "build_time": datetime.now().isoformat(),
                "total_documents": len(documents),
                "document_weights": KnowledgeDocumentBuilder.DOCUMENT_WEIGHTS,
                "documents": documents,
                "stats": self.stats
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"知识库JSON文件已保存: {json_file}")
        
        # 3. 保存统计信息
        stats_file = f"{self.output_dir}/build_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存: {stats_file}")
        
        # 4. 保存按类型分组的文档（便于分析）
        self.save_documents_by_type(documents)
        
        # 5. 生成README文件
        self.generate_readme()
    
    def save_documents_by_type(self, documents: List[Dict[str, Any]]):
        """按文档类型保存文档"""
        docs_by_type = {}
        
        for doc in documents:
            doc_type = doc.get('type', 'unknown')
            if doc_type not in docs_by_type:
                docs_by_type[doc_type] = []
            docs_by_type[doc_type].append(doc)
        
        # 保存每种类型的文档
        for doc_type, type_docs in docs_by_type.items():
            type_file = f"{self.output_dir}/processed/{doc_type}_documents.json"
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": doc_type,
                    "count": len(type_docs),
                    "weight": KnowledgeDocumentBuilder.assign_weight(doc_type),
                    "documents": type_docs[:100]  # 只保存前100个作为样本
                }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已按类型保存文档到 {self.output_dir}/processed/")
    
    def cleanup_batch_files(self):
        """清理批次文件（可选）"""
        batches_dir = f"{self.output_dir}/batches"
        if os.path.exists(batches_dir):
            import shutil
            try:
                shutil.rmtree(batches_dir)
                logger.info(f"批次文件已清理: {batches_dir}")
            except Exception as e:
                logger.warning(f"清理批次文件失败: {e}")
    
    def generate_readme(self):
        """生成README文件"""
        readme_content = f"""# RAG知识库

## 构建信息
"""
        for table, count in self.stats.get("documents_by_table", {}).items():
            readme_content += f"- {table}: {count:,} 个文档\n"
        
        readme_content += "\n## 文档类型统计\n"
        doc_type_stats = sorted(self.stats.get("documents_by_type", {}).items(), 
                              key=lambda x: x[1], reverse=True)
        
        for doc_type, count in doc_type_stats[:20]:  # 显示前20种类型
            weight = KnowledgeDocumentBuilder.assign_weight(doc_type)
            readme_content += f"- {doc_type}: {count:,} 个文档 (权重: {weight})\n"
        
        if len(doc_type_stats) > 20:
            readme_content += f"- ... 等 {len(doc_type_stats)-20} 种其他类型\n"
        
        readme_content += f"""

## 文件说明
1. `knowledge_base.txt` - 纯文本格式，每行一个文档，用于向量化
2. `knowledge_base.json` - 完整JSON格式，包含元数据和权重
3. `build_stats.json` - 构建统计信息
4. `raw/` - 原始数据样本
5. `processed/` - 按类型分组的文档样本

## 优化特性
1. **智能字段选择**: 自动选择最重要字段，避免过长文档
2. **权重分配**: 不同类型文档分配不同检索权重
3. **批量处理**: 支持大数据表分批处理
4. **数据验证**: 自动过滤无效记录
5. **去重处理**: 基于内容去重，避免冗余

## 使用说明
1. 文本文件可用于直接构建向量索引
2. JSON文件可用于详细了解文档结构和元数据
3. 权重信息可用于优化检索排序
4. 可按文档类型进行针对性优化

## 性能指标
- 平均文档长度: {self.calculate_avg_document_length():.0f} 字符
- 文档类型数: {len(self.stats.get("documents_by_type", {}))}
- 数据利用率: {self.calculate_data_utilization():.1%}
"""
        
        with open(f"{self.output_dir}/README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"README文件已生成")
    
    def calculate_avg_document_length(self) -> float:
        """计算平均文档长度"""
        total_length = 0
        total_docs = self.stats.get("total_documents", 0)
        
        if total_docs == 0:
            return 0
        
        # 读取部分文档计算平均长度
        txt_file = f"{self.output_dir}/knowledge_base.txt"
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:1000]  # 取前1000行
                if lines:
                    total_length = sum(len(line.strip()) for line in lines)
                    return total_length / len(lines)
        
        return 300  # 默认值
    
    def calculate_data_utilization(self) -> float:
        """计算数据利用率（有效文档数/总记录数）"""
        total_records = sum(self.stats.get("documents_by_table", {}).values())
        invalid_records = self.stats.get("invalid_records", 0)
        
        if total_records + invalid_records == 0:
            return 0
        
        return total_records / (total_records + invalid_records)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建RAG知识库 - 优化版")
    parser.add_argument('--output-dir', type=str, default='rag_knowledge_base', help='输出目录')
    parser.add_argument('--tables', type=str, nargs='+', 
                       default=['company', 'product', 'price', 'zhaobiao', 'zhongbiao', 'law'], 
                       help='要处理的表名列表')
    parser.add_argument('--batch-size', type=int, default=5000, help='数据库查询批量大小')
    parser.add_argument('--save-batch-size', type=int, default=10000, help='文档保存批量大小')
    parser.add_argument('--test-mode', action='store_true', help='测试模式（只处理少量数据）')
    
    args = parser.parse_args()
    
    # 数据库配置
    DB_CONFIG = {
        "host": "rm-uf63sc2636mu9lk83.mysql.rds.aliyuncs.com",
        "port": 3306,
        "user": "zhaobiao_user",
        "password": "Zyyy20011120",
        "database": "ztb",
        "charset": "utf8mb4"
    }
    
    # 创建构建器
    builder = KnowledgeBaseBuilder(
        DB_CONFIG, 
        batch_size=args.batch_size, 
        output_dir=args.output_dir,
        save_batch_size=args.save_batch_size
    )
    
    # 如果是测试模式，限制数据量
    if args.test_mode:
        logger.info("测试模式启动，只处理前1000条数据")
        # 修改查询限制（通过子类化或修改方法实现）
        # 这里简化处理：只处理前1000条
        original_load = builder.load_table_data_batch
        
        def test_load(table_name):
            df = builder.read_table_data(table_name, limit=1000)
            logger.info(f"测试模式：表 {table_name} 加载 {len(df)} 条记录")
            return df
        
        builder.load_table_data_batch = test_load
    
    # 构建知识库
    start_time = datetime.now()
    result = builder.build_knowledge_base(args.tables)
    end_time = datetime.now()
    
    # 输出统计信息
    stats = result['stats']
    logger.info("\n" + "="*60)
    logger.info("知识库构建完成！")
    logger.info("="*60)
    logger.info(f"总耗时: {(end_time - start_time).total_seconds():.1f} 秒")
    logger.info(f"总文档数: {stats['total_documents']:,}")
    logger.info(f"处理表数: {stats['tables_processed']}")
    logger.info(f"无效记录: {stats.get('invalid_records', 0):,}")
    logger.info(f"重复文档: {stats.get('duplicate_documents', 0):,}")
    
    for table, count in stats.get("documents_by_table", {}).items():
        logger.info(f"{table}: {count:,} 个文档")
    
    # 输出文档类型分布
    logger.info("\n文档类型分布 (Top 10):")
    doc_type_stats = sorted(stats.get("documents_by_type", {}).items(), 
                          key=lambda x: x[1], reverse=True)
    
    for doc_type, count in doc_type_stats[:10]:
        weight = KnowledgeDocumentBuilder.assign_weight(doc_type)
        logger.info(f"  {doc_type}: {count:,} 个文档 (权重: {weight})")
    
    logger.info(f"\n输出目录: {os.path.abspath(builder.output_dir)}")
    logger.info("="*60)


if __name__ == "__main__":
    main()