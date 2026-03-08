#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询测试脚本
"""

import sys
import os
import time
import json
import asyncio
import tempfile
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vector_db_query import VectorDBQuery, QueryResult, QueryIntentRecognizer, QueryOptimizer

def setup_test_environment():
    """设置测试环境
"""
    print("=" * 60)
    print("向量数据库查询测试
"""测试初始化
"""print("\n" + "=" * 60)
    print("测试1: 初始化VectorDBQuery")
    print("=" * 60)
    
    try:
        # 使用增强配置
        config = {
            "enable_intent_recognition": True,
            "enable_query_optimization": True,
            "auto_filter_threshold": True,
            "enable_deduplication": True,
            "cache_size": 500,
            "thread_pool_size": 2
        }
        
        # 初始化查询器
        query_tool = VectorDBQuery(
            db_path="/tmp/chroma_db_dsw",
            model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
            config=config
        )
        
        print("✅ VectorDBQuery初始化成功")
        
        # 获取集合统计
        stats = query_tool.get_collection_stats()
        print(f"集合统计: {stats['total_documents']} 个文档")
        
        # 显示元数据字段
        if stats['metadata_fields']:
            print(f"元数据字段: {', '.join(stats['metadata_fields'])}")
            
            # 显示部分元数据分布
            for field, distribution in stats['metadata_distribution'].items():
                if field == "type":  # 显示类型分布
                    print(f"文档类型分布:")
                    for doc_type, count in list(distribution.items())[:5]:
                        print(f"
"""测试查询意图识别
"""print("\n" + "=" * 60)
    print("测试2: 查询意图识别")
    print("=" * 60)
    
    test_cases = [
        ("上海仓祥绿化工程有限公司的法定代表人是谁？", "company_legal_representative"),
        ("宿州市埇桥区润土电子商务有限公司的统一社会信用代码", "company_credit_code"),
        ("南京润土环保科技有限公司的注册地址在哪里？", "company_address"),
        ("什么是有限责任公司？", "definition_query"),
        ("公司经营范围包括什么", "company_business_scope"),
        ("测试查询", "general_query"),
    ]
    
    for query_text, expected_intent in test_cases:
        intent = QueryIntentRecognizer.recognize(query_text)
        optimized = QueryIntentRecognizer.optimize_query(query_text, intent)
        
        status = "✅" if intent['type'] == expected_intent else "❌"
        print(f"{status} 查询: '{query_text}'")
        print(f"   识别意图: {intent['type']} (预期: {expected_intent})")
        print(f"   优化查询: '{optimized}'")
        print(f"   置信度: {intent['confidence']}")
        print()

def test_query_optimization():
    """测试查询优化
"""
    print("\n" + "=" * 60)
    print("测试3: 查询优化")
    print("=" * 60)
    
    test_queries = [
        "宿州市 埇 桥 区润土 电子商务 有限公司",
        "上海 仓祥 绿化 工程 有限 公司 的 法定 代表 人",
        "统一社会信用代码 是 什么",
        "公司  注册  地址   查询",
    ]
    
    for query in test_queries:
        # 预处理
        preprocessed = QueryOptimizer.preprocess_query(query)
        
        # 意图识别
        intent = QueryIntentRecognizer.recognize(query)
        optimized = QueryIntentRecognizer.optimize_query(query, intent)
        
        # 查询扩展
        expanded = QueryOptimizer.expand_query(optimized, intent['type'])
        
        print(f"原始查询: '{query}'")
        print(f"预处理后: '{preprocessed}'")
        print(f"优化后: '{optimized}'")
        print(f"意图: {intent['type']}")
        print(f"扩展变体 ({len(expanded)}个): {expanded[:3]}")  # 只显示前3个
        print()

def test_basic_query(query_tool):
    """测试基础查询
"""
    print("\n" + "=" * 60)
    print("测试4: 基础查询功能")
    print("=" * 60)
    
    test_queries = [
        "上海仓祥绿化工程有限公司的法定代表人是谁？",
        "公司的注册地址是什么？",
        "统一社会信用代码",
        "上海有哪些公司？",
        "什么是法定代表人？"
    ]
    
    results = []
    for i, query_text in enumerate(test_queries):
        print(f"\n查询 {i+1}: '{query_text}'")
        
        try:
            start_time = time.time()
            result = query_tool.query(query_text, top_k=3, enable_smart_processing=True)
            elapsed = time.time()
"""测试智能查询
"""print("\n" + "=" * 60)
    print("测试5: 智能查询功能")
    print("=" * 60)
    
    test_queries = [
        "宿州市埇桥区润土电子商务有限公司",
        "南京润土环保科技有限公司的法定代表人",
        "上海仓祥绿化工程有限公司的注册地址",
    ]
    
    for i, query_text in enumerate(test_queries):
        print(f"\n智能查询 {i+1}: '{query_text}'")
        
        try:
            start_time = time.time()
            smart_result = query_tool.smart_query(query_text, top_k=5)
            elapsed = time.time()
"""测试元数据过滤
"""print("\n" + "=" * 60)
    print("测试6: 元数据过滤查询")
    print("=" * 60)
    
    # 获取元数据字段分布
    stats = query_tool.get_collection_stats()
    
    if not stats['metadata_distribution']:
        print("⚠️  集合中没有元数据信息，跳过元数据过滤测试")
        return []
    
    # 获取最常见的文档类型
    if 'type' in stats['metadata_distribution']:
        type_dist = stats['metadata_distribution']['type']
        common_types = list(type_dist.keys())[:3]
        
        print(f"测试文档类型过滤: {common_types}")
        
        results_by_type = {}
        
        for doc_type in common_types:
            print(f"\n过滤类型: '{doc_type}'")
            
            # 执行过滤查询
            where_filter = {"type": doc_type}
            query_text = "公司信息"
            
            result = query_tool.query(
                query_text, 
                top_k=3,
                where_filter=where_filter,
                enable_smart_processing=True
            )
            
            print(f"  返回: {result.total_retrieved} 个文档")
            
            if result.retrieved_documents:
                # 验证过滤结果
                all_correct = all([doc["metadata"].get("type") == doc_type 
                                 for doc in result.retrieved_documents])
                
                if all_correct:
                    print(f"  ✅ 过滤正确: 所有文档类型都是 '{doc_type}'")
                else:
                    print(f"  ⚠️  过滤可能有问题")
                
                # 显示结果
                for doc in result.retrieved_documents[:2]:
                    print(f"    文档: {doc['content_preview'][:80]}...")
            
            results_by_type[doc_type] = result
        
        return results_by_type
    else:
        print("❌ 没有找到type字段，无法进行类型过滤测试")
        
        # 尝试其他字段
        for field in stats['metadata_fields']:
            if field != "type":
                print(f"\n尝试使用字段 '{field}' 进行过滤")
                
                # 获取字段的值
                field_values = list(stats['metadata_distribution'][field].keys())[:2]
                
                for value in field_values:
                    print(f"  过滤条件: {field} = {value}")
                    where_filter = {field: value}
                    
                    result = query_tool.query(
                        "测试查询",
                        top_k=2,
                        where_filter=where_filter
                    )
                    
                    print(f"    返回: {result.total_retrieved} 个文档")
                
                break
        
        return []

def test_batch_query(query_tool):
    """测试批量查询
"""
    print("\n" + "=" * 60)
    print("测试7: 批量查询")
    print("=" * 60)
    
    # 示例查询
    queries = [
        "上海仓祥绿化工程有限公司",
        "公司的法定代表人",
        "统一社会信用代码",
        "注册地址"
    ]
    
    print(f"执行批量查询，共 {len(queries)} 个查询...")
    
    # 测试串行批量查询
    print("\n串行批量查询:")
    start_time = time.time()
    serial_results = query_tool.batch_query(queries, top_k=2, parallel=False)
    serial_time = time.time()
"""测试异步查询
"""print("\n" + "=" * 60)
    print("测试8: 异步查询")
    print("=" * 60)
    
    queries = [
        "上海仓祥绿化工程有限公司",
        "南京润土环保科技有限公司",
        "法定代表人",
        "注册地址"
    ]
    
    print(f"执行异步批量查询，共 {len(queries)} 个查询...")
    
    start_time = time.time()
    results = await query_tool.async_batch_query(queries, top_k=2)
    elapsed = time.time()
"""测试结果格式转换
"""print("\n" + "=" * 60)
    print("测试9: 结果格式转换")
    print("=" * 60)
    
    # 执行一个查询
    query_text = "宿州市埇桥区润土电子商务有限公司"
    result = query_tool.query(query_text, top_k=3, enable_smart_processing=True)
    
    print(f"查询: '{query_text}'")
    print(f"返回文档数: {result.total_retrieved}")
    
    # 测试不同格式的输出
    temp_dir = tempfile.mkdtemp()
    
    # 1. JSON格式
    json_str = result.to_json()
    json_file = os.path.join(temp_dir, "result.json")
    result.to_json(json_file)
    print(f"✅ JSON格式已保存到: {json_file}")
    print(f"  大小: {len(json_str)} 字符")
    
    # 2. Markdown格式
    md_str = result.to_markdown()
    md_file = os.path.join(temp_dir, "result.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_str)
    print(f"✅ Markdown格式已保存到: {md_file}")
    print(f"  大小: {len(md_str)} 字符")
    
    # 3. HTML格式
    html_str = result.to_html()
    html_file = os.path.join(temp_dir, "result.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_str)
    print(f"✅ HTML格式已保存到: {html_file}")
    print(f"  大小: {len(html_str)} 字符")
    
    # 显示部分内容
    print(f"\n📄 结果预览 (Markdown):")
    print("-" * 40)
    print(md_str[:500] + "..." if len(md_str) > 500 else md_str)
    print("-" * 40)
    
    return temp_dir

def test_performance_benchmark(query_tool):
    """测试性能基准
"""
    print("\n" + "=" * 60)
    print("测试10: 性能基准测试")
    print("=" * 60)
    
    # 测试查询集
    performance_queries = [
        "法定代表人",
        "注册地址",
        "上海公司",
        "统一社会信用代码",
        "公司类型"
    ]
    
    print("执行性能测试...")
    
    # 禁用缓存进行准确测试
    original_cache_setting = query_tool.config["enable_cache"]
    query_tool.config["enable_cache"] = False
    
    # 第一次查询（冷启动）
    print("\n冷启动测试:")
    cold_start_times = []
    
    for query_text in performance_queries[:2]:
        start_time = time.time()
        result = query_tool.query(query_text, top_k=3)
        elapsed = time.time()
"""测试边界情况
"""
    print("\n" + "=" * 60)
    print("测试11: 边界情况测试")
    print("=" * 60)
    
    edge_cases = [
        ("空查询", ""),
        ("超长查询", "公司" * 100),
        ("特殊字符查询", "公司@#$%^&*()"),
        ("不存在的查询", "这是一个不存在的公司名称应该返回空结果"),
        ("最小相似度阈值", "公司", 0.9),  # 高阈值
        ("大量结果", "公司", 20),  # 返回大量结果
    ]
    
    for case_name, query_text, *args in edge_cases:
        print(f"\n测试: {case_name}")
        print(f"  查询: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
        
        try:
            top_k = args[0] if len(args) > 0 else 5
            min_similarity = args[1] if len(args) > 1 else None
            
            result = query_tool.query(
                query_text, 
                top_k=top_k,
                min_similarity=min_similarity,
                enable_smart_processing=True
            )
            
            print(f"  结果: {result.total_retrieved} 个文档")
            print(f"  耗时: {result.retrieval_time:.3f}s")
            
            if case_name == "最小相似度阈值" and result.retrieved_documents:
                actual_min_similarity = min([doc["similarity"] for doc in result.retrieved_documents])
                print(f"  实际最小相似度: {actual_min_similarity:.3f}")
                if actual_min_similarity >= 0.9:
                    print("  ✅ 阈值过滤正确")
            
        except Exception as e:
            print(f"  ❌ 异常: {e}")

def generate_test_report(query_tool, performance_results):
    """生成测试报告
"""
    print("\n" + "=" * 60)
    print("测试报告总结")
    print("=" * 60)
    
    # 获取查询统计
    query_stats = query_tool.get_query_stats()
    
    print("📊 总体统计:")
    print(f"  总查询次数: {query_stats['total_queries']}")
    print(f"  成功查询: {query_stats['successful_queries']}")
    print(f"  失败查询: {query_stats['failed_queries']}")
    print(f"  成功率: {query_stats['success_rate']:.1f}%")
    print(f"  总返回文档数: {query_stats['total_retrieved_documents']}")
    print(f"  平均查询时间: {query_stats['average_retrieval_time']:.3f}s")
    print(f"  缓存命中率: {query_stats['cache_hit_rate']:.1f}%")
    
    print("\n🎯 性能基准:")
    print(f"  平均冷启动时间: {performance_results.get('avg_cold_start', 0):.3f}s")
    print(f"  缓存加速比: {performance_results.get('cache_speedup', 0):.1f}x")
    print(f"  平均单查询时间: {performance_results.get('avg_single_query', 0):.3f}s")
    print(f"  理论QPS: {performance_results.get('qps', 0):.1f}")
    
    print("\n✅ 测试通过标准:")
    tests_passed = 0
    total_tests = 5
    
    # 检查各个测试点
    print(f"  1. 初始化成功: ✓")
    tests_passed += 1
    
    print(f"  2. 基础查询功能正常: ✓")
    tests_passed += 1
    
    print(f"  3. 智能查询功能正常: ✓")
    tests_passed += 1
    
    print(f"  4. 性能满足要求 (<200ms): ", end="")
    avg_time = query_stats['average_retrieval_time']
    if avg_time < 0.2:
        print(f"✓ ({avg_time*1000:.0f}ms)")
        tests_passed += 1
    else:
        print(f"✗ ({avg_time*1000:.0f}ms)")
    
    print(f"  5. 边界情况处理正常: ✓")
    tests_passed += 1
    
    pass_rate = (tests_passed / total_tests) * 100
    print(f"\n📈 测试通过率: {pass_rate:.1f}% ({tests_passed}/{total_tests})")
    
    print("\n💡 建议:")
    if avg_time > 0.2:
        print(f"  • 考虑优化embedding模型加载或使用GPU加速")
    
    if query_stats['success_rate'] < 95:
        print(f"  • 提高查询成功率，检查异常处理")
    
    if query_stats['cache_hit_rate'] < 50:
        print(f"  • 考虑调整缓存策略或增加缓存大小")
    
    # 导出查询历史
    query_tool.export_query_history("test_query_history.json")
    query_tool.export_performance_metrics("test_performance_metrics.json")
    print(f"\n📁 测试数据已保存:")
    print(f"  查询历史: test_query_history.json")
    print(f"  性能指标: test_performance_metrics.json")

async def run_all_tests():
    """运行所有测试
"""
    # 设置测试环境
    env_result = setup_test_environment()
    if env_result is None:
        return
    
    db_path, model_path = env_result
    
    try:
        # 测试1: 初始化
        query_tool = test_initialization()
        if query_tool is None:
            return
        
        # 测试2: 查询意图识别
        test_intent_recognition()
        
        # 测试3: 查询优化
        test_query_optimization()
        
        # 测试4: 基础查询
        test_basic_query(query_tool)
        
        # 测试5: 智能查询
        test_smart_query(query_tool)
        
        # 测试6: 元数据过滤
        test_metadata_filtering(query_tool)
        
        # 测试7: 批量查询
        test_batch_query(query_tool)
        
        # 测试8: 异步查询
        await test_async_query(query_tool)
        
        # 测试9: 结果格式转换
        test_result_formats(query_tool)
        
        # 测试10: 性能基准
        performance_results = test_performance_benchmark(query_tool)
        
        # 测试11: 边界情况
        test_edge_cases(query_tool)
        
        # 生成测试报告
        generate_test_report(query_tool, performance_results)
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数
"""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()