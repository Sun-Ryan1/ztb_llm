#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChromaDB向量数据库备份脚本
"""

import os
import shutil
import json
import argparse
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s
"""
    备份ChromaDB向量数据库
    
    Args:
        _dir: 
        backup_base_dir: 
        create_timestamp: 
    """
    logger.info("=" * 60)
    logger.info("开始备份ChromaDB向量数据库")
    logger.info("=" * 60)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        logger.error(f"❌ 源目录不存在: {source_dir}")
        logger.info("请检查源目录路径是否正确")
        return False
    
    # 检查源目录是否为空
    if not os.listdir(source_dir):
        logger.error(f"❌ 源目录为空: {source_dir}")
        return False
    
    try:
        # 创建备份基础目录
        os.makedirs(backup_base_dir, exist_ok=True)
        
        # 确定备份目标路径
        if create_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir_name = f"chroma_backup_{timestamp}"
            backup_target = os.path.join(backup_base_dir, backup_dir_name)
        else:
            # 使用固定名称
            backup_target = os.path.join(backup_base_dir, "chroma_db_backup")
            # 如果已存在，添加时间戳后缀
            if os.path.exists(backup_target):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_target = f"{backup_target}_{timestamp}"
        
        logger.info(f"源目录: {source_dir}")
        logger.info(f"备份到: {backup_target}")
        
        # 获取源目录信息
        source_size = get_directory_size(source_dir)
        source_file_count = count_files(source_dir)
        
        logger.info(f"源目录统计:")
        logger.info(f"  文件数量: {source_file_count} 个")
        logger.info(f"  目录大小: {source_size:.2f} MB")
        
        # 列出源目录中的文件
        logger.info("源目录内容:")
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path) / 1024 / 1024  # MB
                logger.info(f"  📄 {item} ({size:.2f} MB)")
            else:
                logger.info(f"  📁 {item}/")
        
        # 执行备份
        logger.info("正在备份...")
        shutil.copytree(source_dir, backup_target)
        
        # 验证备份
        if os.path.exists(backup_target):
            backup_size = get_directory_size(backup_target)
            backup_file_count = count_files(backup_target)
            
            logger.info(f"✅ 备份成功!")
            logger.info(f"备份验证:")
            logger.info(f"  文件数量: {backup_file_count} 个")
            logger.info(f"  目录大小: {backup_size:.2f} MB")
            
            # 保存备份元数据
            metadata = {
                "backup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_dir": source_dir,
                "backup_dir": backup_target,
                "source_size_mb": source_size,
                "backup_size_mb": backup_size,
                "file_count": backup_file_count,
                "vector_count": get_vector_count(source_dir)
            }
            
            metadata_file = os.path.join(backup_target, "backup_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"备份元数据已保存: {metadata_file}")
            
            return True
        else:
            logger.error("❌ 备份失败: 目标目录未创建")
            return False
            
    except Exception as e:
        logger.error(f"❌ 备份过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_directory_size(path: str) -> float:
    """获取目录大小（MB）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # 转换为MB

def count_files(path: str) -> int:
    """统计目录中的文件数量"""
    file_count = 0
    for _, _, filenames in os.walk(path):
        file_count += len(filenames)
    return file_count

def get_vector_count(chroma_dir: str) -> int:
    """尝试获取向量数量"""
    try:
        # 检查是否有统计文件
        stats_file = os.path.join(chroma_dir, "build_statistics.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                return stats.get("total_chunks", 0)
        
        # 检查原始统计文件
        alt_stats_file = os.path.join(os.path.dirname(chroma_dir), "vector_db_statistics.json")
        if os.path.exists(alt_stats_file):
            with open(alt_stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                return stats.get("total_chunks", 0)
    except:
        pass
    
    return 0

def list_backups(backup_base_dir: str):
    """列出所有备份"""
    if not os.path.exists(backup_base_dir):
        logger.error(f"备份目录不存在: {backup_base_dir}")
        return
    
    logger.info(f"备份列表 ({backup_base_dir}):")
    
    backups = []
    for item in os.listdir(backup_base_dir):
        backup_path = os.path.join(backup_base_dir, item)
        if os.path.isdir(backup_path):
            size = get_directory_size(backup_path)
            backups.append((item, size, backup_path))
    
    if not backups:
        logger.info("  没有找到备份")
        return
    
    for i, (name, size, path) in enumerate(sorted(backups), 1):
        # 尝试读取元数据
        metadata_file = os.path.join(path, "backup_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    vector_count = metadata.get("vector_count", "未知")
                    backup_time = metadata.get("backup_time", "未知")
                logger.info(f"{i}. {name} ({size:.1f} MB)")
                logger.info(f"   时间: {backup_time}, 向量数: {vector_count}")
            except:
                logger.info(f"{i}. {name} ({size:.1f} MB)")
        else:
            logger.info(f"{i}. {name} ({size:.1f} MB)")
        
        logger.info(f"   路径: {path}")
        logger.info("")

def restore_backup(backup_dir: str, restore_target: str, overwrite: bool = False):
    """恢复备份"""
    logger.info(f"从 {backup_dir} 恢复到 {restore_target}")
    
    if not os.path.exists(backup_dir):
        logger.error(f"备份目录不存在: {backup_dir}")
        return False
    
    if os.path.exists(restore_target):
        if overwrite:
            logger.warning(f"目标目录已存在，将删除: {restore_target}")
            shutil.rmtree(restore_target)
        else:
            logger.error(f"目标目录已存在: {restore_target}")
            logger.info("使用 --overwrite 参数覆盖")
            return False
    
    try:
        logger.info("正在恢复...")
        shutil.copytree(backup_dir, restore_target)
        logger.info(f"✅ 恢复成功: {restore_target}")
        return True
    except Exception as e:
        logger.error(f"❌ 恢复失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ChromaDB向量数据库备份工具")
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 备份命令
    backup_parser = subparsers.add_parser('backup', help='备份ChromaDB')
    backup_parser.add_argument('--source', type=str, default='/tmp/chroma_db_rag',
                              help='源ChromaDB目录路径')
    backup_parser.add_argument('--target', type=str, 
                              default='./rag_knowledge_base/chroma_backups',
                              help='备份目标目录')
    backup_parser.add_argument('--no-timestamp', action='store_true',
                              help='不使用时间戳命名备份目录')
    
    # 列表命令
    list_parser = subparsers.add_parser('list', help='列出所有备份')
    list_parser.add_argument('--backup-dir', type=str, 
                           default='./rag_knowledge_base/chroma_backups',
                           help='备份目录路径')
    
    # 恢复命令
    restore_parser = subparsers.add_parser('restore', help='恢复备份')
    restore_parser.add_argument('--backup-dir', type=str, required=True,
                               help='要恢复的备份目录路径')
    restore_parser.add_argument('--target', type=str, default='/tmp/chroma_db_restored',
                               help='恢复目标目录')
    restore_parser.add_argument('--overwrite', action='store_true',
                               help='覆盖已存在的目标目录')
    
    args = parser.parse_args()
    
    if args.command == 'backup':
        backup_chromadb(
            source_dir=args.source,
            backup_base_dir=args.target,
            create_timestamp=not args.no_timestamp
        )
    elif args.command == 'list':
        list_backups(args.backup_dir)
    elif args.command == 'restore':
        restore_backup(
            backup_dir=args.backup_dir,
            restore_target=args.target,
            overwrite=args.overwrite
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()