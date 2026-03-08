import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datapreprocessing import DataCollection

def main():
    """测试中国招标投标网的爬取功能
"""
    print("测试中国招标投标网爬取功能")
    print("=" * 50)
    
    # 创建数据收集对象
    collector = DataCollection(output_dir='test_collected_data')
    
    # 测试招标页面爬取
    print("\n测试招标页面爬取...")
    tender_url = 'https://www.cecbid.org.cn/tender/tender/'
    tender_files = collector.crawl_website(tender_url, 'ztb')
    print(f"招标页面爬取完成，共收集{len(tender_files)}个文件")
    
    # 测试中标页面爬取
    print("\n测试中标页面爬取...")
    bid_url = 'https://www.cecbid.org.cn/bid/bid'
    bid_files = collector.crawl_website(bid_url, 'ztb')
    print(f"中标页面爬取完成，共收集{len(bid_files)}个文件")
    
    # 打印总收集数量
    total_files = len(tender_files) + len(bid_files)
    print(f"\n总收集文件数: {total_files}")
    
    if total_files > 0:
        print("✅ 爬取测试成功！")
    else:
        print("❌ 爬取测试失败，未收集到任何文件")

if __name__ == "__main__":
    main()
