#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高级功能交互演示脚本
支持混合检索、查询扩展、重排序等高级功能
"""

import sys
import os
import json
import time
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vector_db_query_advanced import AdvancedVectorDBQuery

class AdvancedQueryDemo:
    """高级查询演示类
"""
    
    def __init__(self):
        self.query_tool = None
        self.current_config = {
            "use_hybrid": True,
            "use_expansion": True,
            "use_reranking": True,
            "top_k": 5
        }
    
    def initialize(self):
        """初始化高级查询器
"""
        print("🚀 初始化高级向量数据库查询器...")
        
        try:
            self.query_tool = AdvancedVectorDBQuery(
                db_path="/tmp/chroma_db_dsw",
                model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3"
            )
            
            # 显示集合信息
            stats = self.query_tool.get_collection_stats()
            print(f"✅ 连接成功! 集合中有 {stats['total_documents']} 个文档")
            
            # 显示高级功能状态
            print(f"\n🎯 高级功能状态:")
            print(f"  混合检索: {'启用' if self.query_tool.config['enable_hybrid_search'] else '禁用'}")
            print(f"  查询扩展: {'启用' if self.query_tool.config['enable_query_expansion'] else '禁用'}")
            print(f"  重排序: {'启用' if self.query_tool.config['enable_reranking'] else '禁用'}")
            print(f"  缓存: {'启用' if self.query_tool.config['enable_cache'] else '禁用'}")
            
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def interactive_query(self):
        """交互式查询
"""
        print("\n" + "=" * 60)
        print("高级交互式查询演示")
        print("=" * 60)
        print("输入查询文本，或使用以下命令:")
        print("  :config
"""执行高级查询
"""print(f"\n📝 查询: '{query_text}'")
        print(f"📋 当前配置: hybrid={self.current_config['use_hybrid']}, "
              f"expansion={self.current_config['use_expansion']}, "
              f"reranking={self.current_config['use_reranking']}, "
              f"top_k={self.current_config['top_k']}")
        
        # 询问是否使用元数据过滤
        use_filter = input("是否使用元数据过滤? (y/n): ").strip().lower() == 'y'
        
        where_filter = None
        if use_filter:
            where_filter = self._get_filter_conditions()
        
        # 执行查询
        start_time = time.time()
        
        result = self.query_tool.query(
            query_text,
            top_k=self.current_config['top_k'],
            where_filter=where_filter,
            use_hybrid=self.current_config['use_hybrid'],
            use_expansion=self.current_config['use_expansion'],
            use_reranking=self.current_config['use_reranking']
        )
        
        query_time = time.time()
"""获取过滤条件
"""
        # ... (与原始版本相同，省略)
        pass
    
    def _handle_command(self, command: str):
        """处理命令
"""
        command = command.lower()
        
        if command == "config":
            self._configure_parameters()
        elif command == "compare":
            self._compare_methods()
        elif command == "expand":
            self._demo_query_expansion()
        elif command == "hybrid":
            self._demo_hybrid_search()
        elif command == "rerank":
            self._demo_reranking()
        elif command == "stats":
            self._show_advanced_stats()
        elif command == "cache":
            self._manage_cache()
        elif command == "batch":
            self._demo_advanced_batch()
        elif command == "history":
            self._show_history()
        elif command == "export":
            self._export_history()
        elif command == "quit":
            print("👋 再见!")
            sys.exit(0)
        else:
            print(f"❌ 未知命令: {command}")
    
    def _configure_parameters(self):
        """配置检索参数
"""
        print("\n⚙️  配置检索参数")
        print("当前配置:")
        print(f"  混合检索: {'启用' if self.current_config['use_hybrid'] else '禁用'}")
        print(f"  查询扩展: {'启用' if self.current_config['use_expansion'] else '禁用'}")
        print(f"  重排序: {'启用' if self.current_config['use_reranking'] else '禁用'}")
        print(f"  返回数量: {self.current_config['top_k']}")
        
        print("\n选择要配置的选项:")
        print("  1. 切换混合检索")
        print("  2. 切换查询扩展")
        print("  3. 切换重排序")
        print("  4. 设置返回数量")
        print("  5. 恢复默认配置")
        print("  0. 取消")
        
        choice = input("请输入选项编号: ").strip()
        
        if choice == "1":
            self.current_config['use_hybrid'] = not self.current_config['use_hybrid']
            print(f"混合检索: {'启用' if self.current_config['use_hybrid'] else '禁用'}")
        elif choice == "2":
            self.current_config['use_expansion'] = not self.current_config['use_expansion']
            print(f"查询扩展: {'启用' if self.current_config['use_expansion'] else '禁用'}")
        elif choice == "3":
            self.current_config['use_reranking'] = not self.current_config['use_reranking']
            print(f"重排序: {'启用' if self.current_config['use_reranking'] else '禁用'}")
        elif choice == "4":
            top_k = input(f"请输入返回数量 (当前: {self.current_config['top_k']}): ").strip()
            if top_k.isdigit():
                self.current_config['top_k'] = int(top_k)
                print(f"返回数量设置为: {self.current_config['top_k']}")
        elif choice == "5":
            self.current_config = {
                "use_hybrid": True,
                "use_expansion": True,
                "use_reranking": True,
                "top_k": 5
            }
            print("已恢复默认配置")
    
    def _compare_methods(self):
        """比较不同检索方法
"""
        print("\n📊 检索方法比较")
        
        query_text = input("请输入查询文本: ").strip()
        if not query_text:
            print("❌ 查询文本不能为空")
            return
        
        top_k = input(f"返回结果数量 (默认 {self.current_config['top_k']}): ").strip()
        top_k = int(top_k) if top_k.isdigit() else self.current_config['top_k']
        
        print(f"\n执行方法比较，查询: '{query_text}'")
        
        try:
            comparison = self.query_tool.compare_retrieval_methods(query_text, top_k=top_k)
            
            print(f"\n✅ 方法比较完成:")
            print("=" * 80)
            
            for method, stats in comparison.items():
                print(f"\n{method.upper()}:")
                print(f"  耗时: {stats['time']:.3f}s")
                print(f"  返回文档: {stats['docs']}")
                print(f"  平均相似度: {stats['avg_similarity']:.3f}")
                print(f"  检索方法: {stats['method']}")
                
                if 'reranked' in stats:
                    print(f"  是否重排序: {stats['reranked']}")
            
            # 找出最佳方法
            best_method = max(comparison.items(), key=lambda x: x[1]['avg_similarity'])
            print(f"\n🎯 最佳方法: {best_method[0]} (相似度: {best_method[1]['avg_similarity']:.3f})")
            
        except Exception as e:
            print(f"❌ 方法比较失败: {e}")
    
    def _demo_query_expansion(self):
        """演示查询扩展
"""
        print("\n🔍 查询扩展演示")
        
        query_text = input("请输入查询文本: ").strip()
        if not query_text:
            print("❌ 查询文本不能为空")
            return
        
        print(f"\n原始查询: '{query_text}'")
        
        # 显示扩展结果
        expansion_result = self.query_tool.query_expander.expand(query_text)
        
        print(f"\n扩展方法: {expansion_result.expansion_method}")
        print(f"扩展耗时: {expansion_result.expansion_time:.3f}s")
        print(f"扩展结果 ({len(expansion_result.expanded_queries)} 个):")
        
        for i, exp_query in enumerate(expansion_result.expanded_queries):
            print(f"  {i+1}. {exp_query}")
        
        # 执行扩展查询
        execute = input("\n是否执行扩展查询? (y/n): ").strip().lower() == 'y'
        if execute:
            top_k = input(f"返回结果数量 (默认 3): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 3
            
            print(f"\n执行扩展查询...")
            
            for i, exp_query in enumerate(expansion_result.expanded_queries):
                print(f"\n扩展查询 {i+1}: '{exp_query}'")
                
                result = self.query_tool.query(
                    exp_query,
                    top_k=top_k,
                    use_hybrid=False,
                    use_expansion=False,
                    use_reranking=False
                )
                
                print(f"  返回: {result.total_retrieved} 个文档")
                print(f"  平均相似度: {result.avg_similarity:.3f}")
                
                if result.retrieved_documents:
                    for doc in result.retrieved_documents[:2]:
                        print(f"    • {doc['content_preview'][:80]}...")
    
    def _demo_hybrid_search(self):
        """演示混合检索
"""
        print("\n🔄 混合检索演示")
        
        query_text = input("请输入查询文本: ").strip()
        if not query_text:
            print("❌ 查询文本不能为空")
            return
        
        top_k = input(f"返回结果数量 (默认 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5
        
        print(f"\n查询: '{query_text}'")
        print(f"返回数量: {top_k}")
        
        # 执行纯向量检索
        print(f"\n1. 纯向量检索:")
        vector_start = time.time()
        vector_result = self.query_tool.query(
            query_text,
            top_k=top_k,
            use_hybrid=False,
            use_expansion=False,
            use_reranking=False
        )
        vector_time = time.time()
"""演示重排序
"""print("\n📈 重排序演示")
        
        query_text = input("请输入查询文本: ").strip()
        if not query_text:
            print("❌ 查询文本不能为空")
            return
        
        top_k = input(f"返回结果数量 (默认 5): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 5
        
        # 先获取混合检索结果
        print(f"\n1. 混合检索结果 (无重排序):")
        normal_result = self.query_tool.query(
            query_text,
            top_k=top_k * 2,  # 获取更多结果用于重排序
            use_hybrid=True,
            use_expansion=False,
            use_reranking=False
        )
        
        print(f"   返回: {normal_result.total_retrieved} 个文档")
        print(f"   平均相似度: {normal_result.avg_similarity:.3f}")
        
        if normal_result.total_retrieved == 0:
            print("❌ 没有检索到文档，无法进行重排序")
            return
        
        # 执行重排序
        print(f"\n2. 重排序后结果:")
        rerank_result = self.query_tool.query(
            query_text,
            top_k=top_k,
            use_hybrid=True,
            use_expansion=False,
            use_reranking=True
        )
        
        print(f"   返回: {rerank_result.total_retrieved} 个文档")
        print(f"   平均相似度: {rerank_result.avg_similarity:.3f}")
        print(f"   是否重排序: {rerank_result.reranked}")
        
        # 显示重排序效果
        if rerank_result.reranked and rerank_result.retrieved_documents:
            print(f"\n📊 重排序效果 (前3个结果):")
            
            for i, doc in enumerate(rerank_result.retrieved_documents[:3]):
                original_score = doc.get('similarity', doc.get('combined_score', 0))
                reranker_score = doc.get('reranker_score', 0)
                score_change = reranker_score
"""显示高级统计信息
"""print("\n📊 高级统计信息:")
        
        # 查询统计
        query_stats = self.query_tool.get_query_stats()
        print(f"\n查询统计:")
        print(f"  总查询数: {query_stats['total_queries']}")
        print(f"  向量检索: {query_stats['vector_queries']}")
        print(f"  混合检索: {query_stats['hybrid_queries']}")
        print(f"  查询扩展: {query_stats['expanded_queries']}")
        print(f"  重排序: {query_stats['reranked_queries']}")
        print(f"  平均响应时间: {query_stats.get('average_retrieval_time', 0):.3f}s")
        
        # 缓存统计
        cache_stats = self.query_tool.query_cache.get_stats()
        print(f"\n缓存统计:")
        print(f"  缓存大小: {cache_stats['size']}")
        print(f"  命中率: {cache_stats['hit_rate']}")
        print(f"  命中次数: {cache_stats['hits']}")
        print(f"  未命中次数: {cache_stats['misses']}")
        print(f"  淘汰次数: {cache_stats['evictions']}")
        
        # 集合统计
        collection_stats = self.query_tool.get_collection_stats()
        print(f"\n集合统计:")
        print(f"  文档总数: {collection_stats['total_documents']}")
        print(f"  元数据字段: {', '.join(collection_stats['metadata_fields'])}")
    
    def _manage_cache(self):
        """缓存管理
"""
        print("\n💾 缓存管理")
        
        cache_stats = self.query_tool.query_cache.get_stats()
        
        print(f"当前缓存状态:")
        print(f"  缓存大小: {cache_stats['size']}")
        print(f"  命中率: {cache_stats['hit_rate']}")
        print(f"  命中次数: {cache_stats['hits']}")
        print(f"  未命中次数: {cache_stats['misses']}")
        
        print("\n选项:")
        print("  1. 清空缓存")
        print("  2. 查看缓存内容")
        print("  3. 导出缓存统计")
        print("  0. 返回")
        
        choice = input("请输入选项编号: ").strip()
        
        if choice == "1":
            confirm = input("确认清空缓存? (y/n): ").strip().lower() == 'y'
            if confirm:
                self.query_tool.clear_cache()
                print("✅ 缓存已清空")
        
        elif choice == "2":
            # 显示部分缓存内容
            print("\n缓存内容 (最近10个):")
            # 注意：这里简化实现，实际可能需要从缓存中获取
            print("  缓存内容查看功能正在开发中...")
        
        elif choice == "3":
            filename = input("输入文件名 (默认: cache_stats.json): ").strip() or "cache_stats.json"
            with open(filename, 'w') as f:
                json.dump(cache_stats, f, indent=2)
            print(f"✅ 缓存统计已导出到 {filename}")
    
    def _demo_advanced_batch(self):
        """演示高级批量查询
"""
        print("\n📦 高级批量查询演示")
        
        # 示例查询
        example_queries = [
            "公司注册地址",
            "法定代表人",
            "统一社会信用代码",
            "公司经营范围",
            "注册资本信息"
        ]
        
        print("示例查询:")
        for i, query in enumerate(example_queries, 1):
            print(f"  {i}. {query}")
        
        use_examples = input("\n使用示例查询? (y/n): ").strip().lower() == 'y'
        
        if use_examples:
            queries = example_queries
        else:
            # 用户自定义
            print("\n输入多个查询 (每行一个，空行结束):")
            queries = []
            while True:
                query = input().strip()
                if not query:
                    break
                queries.append(query)
        
        if not queries:
            print("⚠️  没有查询输入")
            return
        
        # 配置批量查询参数
        print(f"\n批量查询配置:")
        print(f"  查询数量: {len(queries)}")
        
        use_hybrid = input(f"使用混合检索? (y/n, 默认 y): ").strip().lower() != 'n'
        use_expansion = input(f"使用查询扩展? (y/n, 默认 y): ").strip().lower() != 'n'
        use_reranking = input(f"使用重排序? (y/n, 默认 n): ").strip().lower() == 'y'
        top_k = input(f"每个查询返回数量 (默认 3): ").strip()
        top_k = int(top_k) if top_k.isdigit() else 3
        
        print(f"\n开始批量查询...")
        print(f"  配置: hybrid={use_hybrid}, expansion={use_expansion}, reranking={use_reranking}, top_k={top_k}")
        
        start_time = time.time()
        results = self.query_tool.batch_query(
            queries,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_expansion=use_expansion,
            use_reranking=use_reranking
        )
        batch_time = time.time()
"""显示查询历史
"""
        history = self.query_tool.query_history[-10:]  # 最近10条
        
        if not history:
            print("📭 查询历史为空")
            return
        
        print(f"\n📜 最近查询历史 (最近 {len(history)} 条):")
        for i, entry in enumerate(reversed(history), 1):
            print(f"\n  {i}. {entry['timestamp'][11:19]}")
            print(f"     查询: {entry['query'][:50]}...")
            print(f"     方法: {entry.get('retrieval_method', 'vector_search')}")
            print(f"     耗时: {entry['duration']:.3f}s")
            print(f"     文档数: {entry['retrieved_count']}")
            if entry.get('from_cache'):
                print(f"     💾 来自缓存")
    
    def _export_history(self):
        """导出查询历史
"""
        filename = input("输入文件名 (默认: advanced_query_history.json): ").strip() or "advanced_query_history.json"
        
        try:
            self.query_tool.export_query_history(filename)
            print(f"✅ 查询历史已导出到 {filename}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def _save_advanced_result(self, query_text: str, result):
        """保存高级查询结果
"""
        filename = input("输入文件名 (默认: advanced_result.json): ").strip() or "advanced_result.json"
        
        try:
            data = {
                "query": query_text,
                "timestamp": result.query_id,
                "config": self.current_config,
                "result": result.to_dict()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 查询结果已保存到 {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def run(self):
        """运行演示
"""
        print("=" * 60)
        print("高级向量数据库查询演示")
        print("=" * 60)
        
        # 初始化
        if not self.initialize():
            return
        
        # 交互式查询
        self.interactive_query()

def main():
    """主函数
"""
    try:
        demo = AdvancedQueryDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
