#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高级功能测试脚本
测试混合检索、查询扩展、重排序等高级功能
"""

import sys
import os
import time
import json
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vector_db_query_advanced import AdvancedVectorDBQuery

def test_advanced_initialization():
    """测试高级查询器初始化
"""
    print("\n" + "=" * 60)
    print("测试1: 高级查询器初始化")
    print("=" * 60)
    
    try:
        # 初始化高级查询器
        query_tool = AdvancedVectorDBQuery(
            db_path="/tmp/chroma_db_dsw",
            model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
        )
        
        print("✅ AdvancedVectorDBQuery初始化成功")
        
        # 获取集合统计
        stats = query_tool.get_collection_stats()
        print(f"集合统计: {stats['total_documents']} 个文档")
        
        # 检查高级功能配置
        print(f"\n高级功能配置:")
        print(f"  混合检索: {'启用' if query_tool.config['enable_hybrid_search'] else '禁用'}")
        print(f"  查询扩展: {'启用' if query_tool.config['enable_query_expansion'] else '禁用'}")
        print(f"  重排序: {'启用' if query_tool.config['enable_reranking'] else '禁用'}")
        
        # 检查组件初始化
        print(f"\n组件初始化状态:")
        print(f"  混合检索器: {'✓' if query_tool.hybrid_retriever else '✗'}")
        print(f"  查询扩展器: {'✓' if query_tool.query_expander else '✗'}")
        print(f"  重排序器: {'✓' if query_tool.reranker.model else '✗'}")
        
        return query_tool
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_query_expansion(query_tool):
    """测试查询扩展功能
"""
    print("\n" + "=" * 60)
    print("测试2: 查询扩展功能")
    print("=" * 60)
    
    test_queries = [
        "公司注册地址",
        "法定代表人信息",
        "统一社会信用代码查询",
        "上海仓祥绿化工程有限公司"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n测试查询 {i+1}: '{query}'")
        
        try:
            # 测试查询扩展器
            expansion_result = query_tool.query_expander.expand(query)
            
            print(f"  扩展方法: {expansion_result.expansion_method}")
            print(f"  扩展耗时: {expansion_result.expansion_time:.3f}s")
            print(f"  扩展结果 ({len(expansion_result.expanded_queries)} 个):")
            
            for j, exp_query in enumerate(expansion_result.expanded_queries):
                print(f"    {j+1}. {exp_query}")
            
            # 测试带扩展的查询
            print(f"\n  带扩展的查询测试:")
            result = query_tool.query(
                query, 
                top_k=3,
                use_expansion=True,
                use_hybrid=False,
                use_reranking=False
            )
            
            print(f"    检索方法: {result.retrieval_method}")
            print(f"    返回文档: {result.total_retrieved}")
            print(f"    平均相似度: {result.avg_similarity:.3f}")
            
        except Exception as e:
            print(f"❌ 查询扩展测试失败: {e}")

def test_hybrid_search(query_tool):
    """测试混合检索功能
"""
    print("\n" + "=" * 60)
    print("测试3: 混合检索功能")
    print("=" * 60)
    
    test_queries = [
        "上海公司的注册信息",
        "法定代表人变更",
        "公司注册资本",
        "绿化工程公司的经营范围"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n测试查询 {i+1}: '{query}'")
        
        try:
            # 1. 纯向量检索
            vector_start = time.time()
            vector_result = query_tool.query(
                query, top_k=5,
                use_hybrid=False,
                use_expansion=False,
                use_reranking=False
            )
            vector_time = time.time()
"""测试重排序功能
"""print("\n" + "=" * 60)
    print("测试4: 重排序功能")
    print("=" * 60)
    
    test_queries = [
        "公司的法定代表人是谁",
        "注册地址在什么地方",
        "经营范围包括哪些内容"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n测试查询 {i+1}: '{query}'")
        
        try:
            # 1. 无重排序
            normal_start = time.time()
            normal_result = query_tool.query(
                query, top_k=5,
                use_hybrid=True,
                use_expansion=False,
                use_reranking=False
            )
            normal_time = time.time()
"""测试高级缓存功能
"""print("\n" + "=" * 60)
    print("测试5: 高级缓存功能")
    print("=" * 60)
    
    test_query = "上海仓祥绿化工程有限公司"
    
    print(f"测试查询: '{test_query}'")
    
    try:
        # 第一次查询（缓存未命中）
        print(f"\n第一次查询 (缓存未命中):")
        start_time = time.time()
        result1 = query_tool.query(
            test_query, top_k=5,
            use_hybrid=True,
            use_expansion=True,
            use_reranking=True
        )
        time1 = time.time()
"""测试不同检索方法的比较
"""print("\n" + "=" * 60)
    print("测试6: 检索方法比较")
    print("=" * 60)
    
    test_queries = [
        ("简单查询", "公司"),
        ("精确查询", "上海仓祥绿化工程有限公司法定代表人"),
        ("模糊查询", "绿化工程公司的注册信息")
    ]
    
    for query_name, query_text in test_queries:
        print(f"\n测试: {query_name}")
        print(f"  查询: '{query_text}'")
        
        try:
            comparison = query_tool.compare_retrieval_methods(query_text, top_k=5)
            
            print(f"\n  方法比较结果:")
            for method, stats in comparison.items():
                print(f"    {method}:")
                print(f"      耗时: {stats['time']:.3f}s")
                print(f"      文档数: {stats['docs']}")
                print(f"      平均相似度: {stats['avg_similarity']:.3f}")
                print(f"      方法: {stats['method']}")
            
            # 找出最佳方法
            best_method = max(comparison.items(), key=lambda x: x[1]['avg_similarity'])
            print(f"\n  最佳方法: {best_method[0]} (相似度: {best_method[1]['avg_similarity']:.3f})")
            
        except Exception as e:
            print(f"❌ 方法比较测试失败: {e}")

def test_advanced_batch_query(query_tool):
    """测试高级批量查询
"""
    print("\n" + "=" * 60)
    print("测试7: 高级批量查询")
    print("=" * 60)
    
    batch_queries = [
        "公司注册地址",
        "法定代表人",
        "统一社会信用代码",
        "经营范围",
        "注册资本"
    ]
    
    print(f"批量查询 {len(batch_queries)} 个查询:")
    for i, query in enumerate(batch_queries):
        print(f"  {i+1}. {query}")
    
    try:
        start_time = time.time()
        
        # 执行批量查询（使用所有高级功能）
        results = query_tool.batch_query(
            batch_queries,
            top_k=3,
            use_hybrid=True,
            use_expansion=True,
            use_reranking=False  # 批量查询时关闭重排序以减少计算量
        )
        
        batch_time = time.time()
"""生成高级功能测试报告
"""
    print("\n" + "=" * 60)
    print("高级功能测试报告")
    print("=" * 60)
    
    # 获取查询统计
    query_stats = query_tool.get_query_stats()
    cache_stats = query_tool.query_cache.get_stats()
    
    print("📊 高级功能统计:")
    print(f"  总查询次数: {query_stats['total_queries']}")
    print(f"  向量检索次数: {query_stats['vector_queries']}")
    print(f"  混合检索次数: {query_stats['hybrid_queries']}")
    print(f"  查询扩展次数: {query_stats['expanded_queries']}")
    print(f"  重排序次数: {query_stats['reranked_queries']}")
    
    print(f"\n🎯 缓存性能:")
    print(f"  缓存命中率: {cache_stats['hit_rate']}")
    print(f"  缓存命中次数: {query_stats['cache_hits']}")
    print(f"  缓存未命中次数: {query_stats['cache_misses']}")
    
    print(f"\n⏱️  性能指标:")
    avg_time = query_stats.get('average_retrieval_time', 0)
    print(f"  平均查询时间: {avg_time*1000:.0f}ms")
    
    # 导出查询历史
    query_tool.export_query_history("advanced_query_history.json")
    print(f"\n📁 查询历史已保存到: advanced_query_history.json")
    
    # 导出缓存统计
    with open("cache_stats.json", "w") as f:
        json.dump(cache_stats, f, indent=2)
    print(f"📁 缓存统计已保存到: cache_stats.json")

def main():
    """主测试函数
"""
    print("=" * 60)
    print("向量数据库高级功能测试")
    print("=" * 60)
    
    try:
        # 测试1: 初始化
        query_tool = test_advanced_initialization()
        if query_tool is None:
            return
        
        # 测试2: 查询扩展
        test_query_expansion(query_tool)
        
        # 测试3: 混合检索
        test_hybrid_search(query_tool)
        
        # 测试4: 重排序
        test_reranking(query_tool)
        
        # 测试5: 高级缓存
        test_advanced_cache(query_tool)
        
        # 测试6: 方法比较
        test_method_comparison(query_tool)
        
        # 测试7: 批量查询
        test_advanced_batch_query(query_tool)
        
        # 生成报告
        generate_advanced_report(query_tool)
        
        print("\n" + "=" * 60)
        print("✅ 第二阶段高级功能测试完成!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
