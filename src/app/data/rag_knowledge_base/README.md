# RAG知识库

## 构建信息
- 构建时间: 2026-02-14 19:09:02
- 总文档数: 187,091
- 处理表数: 6
- 无效记录: 65
- 重复文档: 642

## 表统计
- company: 162,065 个文档
- product: 10,877 个文档
- price: 7,292 个文档
- zhaobiao: 3,456 个文档
- zhongbiao: 3,305 个文档
- law: 96 个文档

## 文档类型统计
- company_location: 64,826 个文档 (权重: 0.9)
- company_full: 32,413 个文档 (权重: 1.0)
- company_legal_rep: 32,413 个文档 (权重: 0.9)
- company_credit_code: 32,413 个文档 (权重: 0.8)
- product_supplier: 3,772 个文档 (权重: 0.9)
- product_full: 1,890 个文档 (权重: 1.0)
- product_keyword: 1,889 个文档 (权重: 0.7)
- price_full: 1,825 个文档 (权重: 1.0)
- keyword_price: 1,825 个文档 (权重: 0.7)
- supplier_price: 1,821 个文档 (权重: 0.9)
- price_structured: 1,821 个文档 (权重: 0.8)
- product_price: 1,820 个文档 (权重: 0.9)
- supplier_address: 1,506 个文档 (权重: 0.8)
- zhaobiao_full: 1,157 个文档 (权重: 1.0)
- zhaobiao_buyer: 1,151 个文档 (权重: 0.9)
- zhaobiao_supplier: 1,148 个文档 (权重: 0.9)
- zhongbiao_full: 1,113 个文档 (权重: 1.0)
- zhongbiao_supplier: 1,101 个文档 (权重: 0.9)
- zhongbiao_buyer: 1,091 个文档 (权重: 0.9)
- law_full: 28 个文档 (权重: 1.0)
- ... 等 3 种其他类型


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
- 平均文档长度: 49 字符
- 文档类型数: 23
- 数据利用率: 100.0%
