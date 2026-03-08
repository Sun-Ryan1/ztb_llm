#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量数据库查询演示脚本
"""

import sys
import os
import json
import time
import tempfile
import webbrowser
from pathlib import Path
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vector_db_query import VectorDBQuery

class QueryDemo:
    """增强版查询演示类
"""
    
    def __init__(self):
        self.query_tool = None
        self.current_result = None
        self.session_history = []
    
    def initialize(self):
        """初始化查询器
"""
        print("🚀 初始化增强版向量数据库查询器...")
        
        try:
            # 使用增强配置
            config = {
                "enable_intent_recognition": True,
                "enable_query_optimization": True,
                "auto_filter_threshold": True,
                "enable_deduplication": True,
                "enable_grouping": True,
                "cache_size": 500,
                "thread_pool_size": 2,
                "default_top_k": 10
            }
            
            self.query_tool = VectorDBQuery(
                db_path="/tmp/chroma_db_dsw",
                model_path="/mnt/workspace/data/modelscope/cache/bge-m3/BAAI/bge-m3",
                config=config
            )
            
            # 显示集合信息
            stats = self.query_tool.get_collection_stats()
            print(f"✅ 连接成功!")
            print(f"   集合文档数: {stats['total_documents']}")
            print(f"   元数据字段: {', '.join(stats['metadata_fields'][:5])}")
            
            # 显示配置
            print(f"   当前配置:")
            for key, value in list(self.query_tool.config.items())[:5]:
                print(f"
"""交互式查询
"""print("\n" + "=" * 60)
        print("增强版交互式查询演示")
        print("=" * 60)
        print("输入查询文本，或使用以下命令:")
        print("  :smart
"""执行标准查询
"""print(f"\n📝 标准查询: '{query_text}'")
        
        # 询问是否使用过滤
        use_filter = input("是否使用元数据过滤? (y/n, 默认:n): ").strip().lower() == 'y'
        
        where_filter = None
        if use_filter:
            where_filter = self._get_filter_conditions()
        
        # 询问返回数量
        top_k_str = input(f"返回结果数量 (默认 {self.query_tool.config['default_top_k']}): ").strip()
        top_k = int(top_k_str) if top_k_str else self.query_tool.config['default_top_k']
        
        # 询问最小相似度
        min_sim_str = input("最小相似度阈值 (默认:自动): ").strip()
        min_similarity = float(min_sim_str) if min_sim_str else None
        
        # 执行查询
        start_time = time.time()
        result = self.query_tool.query(
            query_text, 
            top_k=top_k, 
            where_filter=where_filter,
            min_similarity=min_similarity,
            enable_smart_processing=True
        )
        elapsed = time.time()
"""执行智能查询
"""print(f"\n🎯 智能查询: '{query_text}'")
        
        # 询问返回数量
        top_k_str = input(f"返回结果数量 (默认 5): ").strip()
        top_k = int(top_k_str) if top_k_str else 5
        
        # 执行智能查询
        start_time = time.time()
        smart_result = self.query_tool.smart_query(query_text, top_k=top_k)
        elapsed = time.time()
"""执行增强查询
"""print(f"\n🚀 增强查询: '{query_text}'")
        
        # 询问返回数量
        top_k_str = input(f"返回结果数量 (默认 {self.query_tool.config['default_top_k']}): ").strip()
        top_k = int(top_k_str) if top_k_str else self.query_tool.config['default_top_k']
        
        # 执行增强查询
        start_time = time.time()
        result = self.query_tool.query(
            query_text, 
            top_k=top_k,
            enable_smart_processing=True
        )
        elapsed = time.time()
"""显示查询结果
"""print(f"\n✅ 查询完成!")
        print(f"   耗时: {elapsed:.3f}s (向量化: {result.embedding_time:.3f}s, 查询: {result.query_time:.3f}s)")
        print(f"   返回: {result.total_retrieved} 个文档")
        print(f"   平均相似度: {result.avg_similarity:.3f}")
        
        if result.intent_type:
            print(f"   识别意图: {result.intent_type}")
        
        if result.optimized_query and result.optimized_query != result.query:
            print(f"   优化查询: {result.optimized_query}")
        
        if result.retrieved_documents:
            print("\n📄 检索到的文档:")
            for i, doc in enumerate(result.retrieved_documents[:10]):  # 只显示前10个
                print(f"\n  [{i+1}] 相似度: {doc['similarity']:.3f}")
                print(f"      内容: {doc['content_preview']}")
                
                # 显示元数据
                if doc['metadata']:
                    metadata_items = list(doc['metadata'].items())[:3]
                    metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata_items])
                    print(f"      元数据: {metadata_str}")
        
        # 询问后续操作
        self._post_query_options(result)
    
    def _display_smart_result(self, smart_result, elapsed: float):
        """显示智能查询结果
"""
        print(f"\n🎯 智能查询完成!")
        print(f"   耗时: {elapsed:.3f}s")
        print(f"   识别意图: {smart_result.get('intent_type', '未知')}")
        
        if smart_result.get('optimized_query'):
            print(f"   优化查询: {smart_result['optimized_query']}")
        
        stats = smart_result['statistics']
        print(f"   总文档数: {stats['total_documents']}")
        print(f"   发现公司: {stats['unique_companies']} 个")
        
        if smart_result['companies']:
            print("\n🏢 公司信息摘要:")
            for i, company in enumerate(smart_result['companies'][:5]):  # 只显示前5个
                print(f"\n  [{i+1}] {company['company_name']}")
                print(f"      文档数: {company['document_count']}")
                print(f"      最大相似度: {company['max_similarity']:.3f}")
                print(f"      信息类型: {', '.join(company['info_types'])}")
                
                if company['key_info']:
                    print(f"      关键信息:")
                    for key, value in company['key_info'].items():
                        print(f"        {key}: {value}")
        
        # 询问是否查看原始结果
        view_raw = input("\n是否查看原始查询结果? (y/n): ").strip().lower() == 'y'
        if view_raw and 'raw_result' in smart_result:
            raw_result = smart_result['raw_result']
            if isinstance(raw_result, dict) and 'retrieved_documents' in raw_result:
                print(f"\n📄 原始结果 (前3个):")
                for i, doc in enumerate(raw_result['retrieved_documents'][:3]):
                    print(f"  [{i+1}] 相似度: {doc['similarity']:.3f}")
                    print(f"      内容: {doc['content_preview']}")
    
    def _display_enhanced_result(self, result, elapsed: float):
        """显示增强查询结果
"""
        self._display_query_result(result, elapsed)
        
        # 额外分析
        if result.retrieved_documents:
            # 按相似度分组
            high_sim = [d for d in result.retrieved_documents if d['similarity'] > 0.7]
            med_sim = [d for d in result.retrieved_documents if 0.4 <= d['similarity'] <= 0.7]
            low_sim = [d for d in result.retrieved_documents if d['similarity'] < 0.4]
            
            print(f"\n📊 相似度分析:")
            print(f"   高相似度 (>0.7): {len(high_sim)} 个文档")
            print(f"   中相似度 (0.4-0.7): {len(med_sim)} 个文档")
            print(f"   低相似度 (<0.4): {len(low_sim)} 个文档")
            
            # 按类型分组
            type_count = {}
            for doc in result.retrieved_documents:
                doc_type = doc['metadata'].get('type', '未知')
                type_count[doc_type] = type_count.get(doc_type, 0) + 1
            
            if type_count:
                print(f"\n📋 文档类型分布:")
                for doc_type, count in list(type_count.items())[:5]:
                    print(f"   {doc_type}: {count} 个")
    
    def _get_filter_conditions(self) -> dict:
        """获取过滤条件
"""
        print("\n🔧 设置过滤条件")
        
        # 获取可用的元数据字段
        stats = self.query_tool.get_collection_stats()
        if not stats['metadata_fields']:
            print("⚠️  集合中没有元数据字段")
            return {}
        
        print(f"可用字段: {', '.join(stats['metadata_fields'][:10])}")
        
        where_filter = {}
        while True:
            field = input("输入字段名 (或按回车结束): ").strip()
            if not field:
                break
            
            if field not in stats['metadata_fields']:
                print(f"⚠️  字段 '{field}' 不存在，跳过")
                continue
            
            value = input(f"输入 {field} 的值: ").strip()
            if value:
                # 尝试转换数值类型
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                
                where_filter[field] = value
                print(f"添加条件: {field} = {value}")
        
        return where_filter
    
    def _post_query_options(self, result):
        """查询后操作选项
"""
        print("\n📋 后续操作:")
        print("  1. 保存结果到文件")
        print("  2. 导出为不同格式")
        print("  3. 在浏览器中查看")
        print("  4. 继续查询")
        
        choice = input("选择操作 (1-4, 默认:4): ").strip()
        
        if choice == "1":
            self._save_query_result(result)
        elif choice == "2":
            self._export_result_formats(result)
        elif choice == "3":
            self._open_in_browser(result)
    
    def _save_query_result(self, result):
        """保存查询结果
"""
        filename = input("输入文件名 (默认: query_result.json): ").strip() or "query_result.json"
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            result.to_json(filename)
            print(f"✅ 查询结果已保存到 {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def _export_result_formats(self, result):
        """导出为不同格式
"""
        print("\n📤 选择导出格式:")
        print("  1. JSON (完整数据)")
        print("  2. Markdown (可读格式)")
        print("  3. HTML (网页格式)")
        print("  4. 全部格式")
        
        choice = input("选择格式 (1-4): ").strip()
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = f"query_exports_{timestamp}"
        os.makedirs(export_dir, exist_ok=True)
        
        try:
            if choice in ["1", "4"]:
                json_file = os.path.join(export_dir, "result.json")
                result.to_json(json_file)
                print(f"✅ JSON 格式: {json_file}")
            
            if choice in ["2", "4"]:
                md_file = os.path.join(export_dir, "result.md")
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(result.to_markdown())
                print(f"✅ Markdown 格式: {md_file}")
            
            if choice in ["3", "4"]:
                html_file = os.path.join(export_dir, "result.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(result.to_html())
                print(f"✅ HTML 格式: {html_file}")
            
            print(f"\n📁 所有文件已保存到: {os.path.abspath(export_dir)}")
            
            # 询问是否打开目录
            open_dir = input("是否打开导出目录? (y/n): ").strip().lower() == 'y'
            if open_dir and sys.platform == "darwin":  # macOS
                os.system(f"open {export_dir}")
            elif open_dir and sys.platform == "win32":  # Windows
                os.system(f"start {export_dir}")
            elif open_dir and sys.platform == "linux":  # Linux
                os.system(f"xdg-open {export_dir}")
                
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def _open_in_browser(self, result):
        """在浏览器中打开结果
"""
        try:
            # 创建临时HTML文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
            temp_file.write(result.to_html())
            temp_file.close()
            
            # 在浏览器中打开
            webbrowser.open(f"file://{temp_file.name}")
            print(f"✅ 已在浏览器中打开结果")
            
            # 询问是否保留文件
            keep = input("是否保留HTML文件? (y/n): ").strip().lower() == 'y'
            if not keep:
                os.unlink(temp_file.name)
            else:
                print(f"文件保存在: {temp_file.name}")
                
        except Exception as e:
            print(f"❌ 打开浏览器失败: {e}")
    
    def _handle_command(self, command: str):
        """处理命令
"""
        command = command.lower()
        
        if command == "smart":
            self._demo_smart_query()
        elif command == "filter":
            self._demo_filtering()
        elif command == "batch":
            self._demo_batch_query()
        elif command == "async":
            self._demo_async_query()
        elif command == "history":
            self._show_history()
        elif command == "stats":
            self._show_stats()
        elif command == "export":
            self._export_session()
        elif command == "open":
            self._open_current_result()
        elif command == "config":
            self._manage_config()
        elif command == "clear":
            self._clear_cache()
        elif command == "help":
            self._show_help()
        elif command == "quit":
            print("👋 再见!")
            sys.exit(0)
        else:
            print(f"❌ 未知命令: {command}")
    
    def _demo_smart_query(self):
        """演示智能查询功能
"""
        print("\n🎯 智能查询演示")
        
        # 示例查询
        examples = [
            "宿州市埇桥区润土电子商务有限公司",
            "南京润土环保科技有限公司的法定代表人",
            "上海仓祥绿化工程有限公司的注册地址",
        ]
        
        print("示例查询:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        
        use_example = input("\n使用示例查询? (y/n): ").strip().lower() == 'y'
        
        if use_example:
            choice = input(f"选择查询 (1-{len(examples)}): ").strip()
            try:
                idx = int(choice)
"""演示过滤功能
"""print("\n🔧 元数据过滤演示")
        
        # 获取集合统计
        stats = self.query_tool.get_collection_stats()
        
        if 'type' in stats['metadata_distribution']:
            print("\n文档类型分布:")
            type_dist = stats['metadata_distribution']['type']
            for doc_type, count in list(type_dist.items())[:10]:
                print(f"  {doc_type}: {count} 个文档")
            
            # 演示过滤
            print("\n🎯 过滤演示:")
            query_text = input("输入查询文本 (默认: 公司信息): ").strip() or "公司信息"
            
            for doc_type in list(type_dist.keys())[:3]:
                print(f"\n过滤类型: '{doc_type}'")
                result = self.query_tool.query(
                    query_text,
                    top_k=2,
                    where_filter={"type": doc_type},
                    enable_smart_processing=True
                )
                
                print(f"  返回: {result.total_retrieved} 个文档")
                if result.retrieved_documents:
                    for doc in result.retrieved_documents:
                        print(f"    • {doc['content_preview'][:80]}...")
    
    def _demo_batch_query(self):
        """演示批量查询
"""
        print("\n📦 批量查询演示")
        
        # 示例查询
        example_queries = [
            "上海仓祥绿化工程有限公司",
            "公司的法定代表人",
            "统一社会信用代码",
            "注册地址"
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
        
        print(f"\n执行批量查询，共 {len(queries)} 个查询...")
        
        # 选择模式
        print("\n选择批量查询模式:")
        print("  1. 串行查询")
        print("  2. 并行查询")
        mode = input("选择模式 (1/2, 默认:2): ").strip()
        parallel = mode != "1"
        
        start_time = time.time()
        results = self.query_tool.batch_query(queries, top_k=2, parallel=parallel)
        elapsed = time.time()
"""异步查询演示实现
"""print("\n⚡ 异步批量查询演示")
        
        queries = [
            "上海仓祥绿化工程有限公司",
            "南京润土环保科技有限公司",
            "法定代表人",
            "注册地址"
        ]
        
        print(f"执行异步批量查询，共 {len(queries)} 个查询...")
        
        import asyncio
        start_time = time.time()
        results = await self.query_tool.async_batch_query(queries, top_k=2)
        elapsed = time.time()
"""演示异步查询
"""import asyncio
        asyncio.run(self._demo_async_query_impl())
    
    def _show_history(self):
        """显示查询历史
"""
        print("\n📜 查询历史:")
        
        # 系统历史
        history = self.query_tool.query_history[-10:]  # 最近10条
        
        if history:
            print("系统查询历史 (最近10条):")
            for i, entry in enumerate(reversed(history), 1):
                print(f"\n  {i}. {entry['timestamp'][11:19]}")
                print(f"     查询: {entry['query'][:50]}...")
                print(f"     耗时: {entry['duration']:.3f}s")
                print(f"     文档数: {entry['retrieved_count']}")
                if entry.get('from_cache'):
                    print(f"     💾 来自缓存")
        
        # 会话历史
        if self.session_history:
            print(f"\n当前会话历史 ({len(self.session_history)} 条):")
            for i, entry in enumerate(self.session_history[-5:], 1):  # 最近5条
                print(f"\n  {i}. {entry['timestamp'][11:19]}")
                print(f"     查询: {entry['query'][:50]}...")
                print(f"     模式: {entry['mode']}")
                if 'result' in entry and isinstance(entry['result'], dict):
                    if 'total_retrieved' in entry['result']:
                        print(f"     文档数: {entry['result']['total_retrieved']}")
        
        if not history and not self.session_history:
            print("📭 查询历史为空")
    
    def _show_stats(self):
        """显示统计信息
"""
        print("\n📊 系统统计信息:")
        
        # 查询统计
        query_stats = self.query_tool.get_query_stats()
        print(f"\n查询统计:")
        print(f"  总查询数: {query_stats['total_queries']}")
        print(f"  成功率: {query_stats['success_rate']:.1f}%")
        print(f"  平均响应时间: {query_stats['average_retrieval_time']:.3f}s")
        print(f"  缓存命中率: {query_stats['cache_hit_rate']:.1f}%")
        print(f"  缓存大小: {query_stats['cache_size']}")
        
        # 集合统计
        collection_stats = self.query_tool.get_collection_stats()
        print(f"\n集合统计:")
        print(f"  文档总数: {collection_stats['total_documents']}")
        print(f"  元数据字段: {', '.join(collection_stats['metadata_fields'][:10])}")
        
        # 性能指标
        if hasattr(self.query_tool, 'performance_metrics'):
            perf_count = len(self.query_tool.performance_metrics)
            if perf_count > 0:
                print(f"\n性能监控:")
                print(f"  记录的性能指标数: {perf_count}")
                
                # 显示最近的性能指标
                recent_times = [m['execution_time'] for m in self.query_tool.performance_metrics[-10:] if 'execution_time' in m]
                if recent_times:
                    avg_time = sum(recent_times) / len(recent_times)
                    print(f"  最近10次平均执行时间: {avg_time:.3f}s")
    
    def _export_session(self):
        """导出会话历史
"""
        filename = input("输入文件名 (默认: session_history.json): ").strip() or "session_history.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.session_history, f, ensure_ascii=False, indent=2)
            print(f"✅ 会话历史已导出到 {filename}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def _open_current_result(self):
        """在当前浏览器中打开结果
"""
        if self.current_result:
            self._open_in_browser(self.current_result)
        else:
            print("⚠️  没有当前查询结果，请先执行查询")
    
    def _manage_config(self):
        """管理配置
"""
        print("\n⚙️  当前配置:")
        for key, value in self.query_tool.config.items():
            print(f"  {key}: {value}")
        
        print("\n选择操作:")
        print("  1. 修改配置")
        print("  2. 重置配置")
        print("  3. 保存配置")
        
        choice = input("选择操作 (1-3, 默认:取消): ").strip()
        
        if choice == "1":
            self._modify_config()
        elif choice == "2":
            self._reset_config()
        elif choice == "3":
            self._save_config()
    
    def _modify_config(self):
        """修改配置
"""
        print("\n修改配置 (输入 key=value 格式，空行结束):")
        print("可用配置项:")
        for key in self.query_tool.config.keys():
            print(f"  {key}")
        
        while True:
            entry = input().strip()
            if not entry:
                break
            
            if '=' in entry:
                key, value = entry.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key in self.query_tool.config:
                    # 尝试转换值类型
                    try:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                        
                        self.query_tool.config[key] = value
                        print(f"✅ 配置已更新: {key} = {value}")
                    except Exception as e:
                        print(f"❌ 更新失败: {e}")
                else:
                    print(f"❌ 未知配置项: {key}")
    
    def _reset_config(self):
        """重置配置
"""
        confirm = input("确认重置配置? (y/n): ").strip().lower() == 'y'
        if confirm:
            default_config = {
                "enable_intent_recognition": True,
                "enable_query_optimization": True,
                "auto_filter_threshold": True,
                "enable_deduplication": True,
                "enable_grouping": True,
                "cache_size": 500,
                "thread_pool_size": 2,
                "default_top_k": 10
            }
            self.query_tool.config.update(default_config)
            print("✅ 配置已重置为默认值")
    
    def _save_config(self):
        """保存配置
"""
        filename = input("输入配置文件名 (默认: query_config.json): ").strip() or "query_config.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.query_tool.config, f, ensure_ascii=False, indent=2)
            print(f"✅ 配置已保存到 {filename}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def _clear_cache(self):
        """清空缓存
"""
        confirm = input("确认清空缓存? (y/n): ").strip().lower() == 'y'
        if confirm:
            self.query_tool.clear_cache()
            print("✅ 缓存已清空")
    
    def _show_help(self):
        """显示帮助
"""
        print("\n📖 帮助信息:")
        print("=" * 40)
        print("查询模式:")
        print("  1. 标准查询
"""运行演示
"""
        print("=" * 60)
        print("增强版向量数据库查询演示")
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
        demo = QueryDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()