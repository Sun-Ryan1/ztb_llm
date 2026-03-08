#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试百度爱采购爬虫的地址提取功能
"""

from product_price_crawler import ProductPriceCrawler

def test_address_extraction():
    """测试地址提取功能
"""
    # 创建爬虫实例
    crawler = ProductPriceCrawler()
    
    # 用户提供的示例URL
    test_url = "https://b2b.baidu.com/land?url=https%3A%2F%2Fb2bwork.baidu.com%2Fland%3Flid%3D1796474319961531401&query=%E5%8F%89%E8%BD%A6&lattr=ot%2C1998.00%E5%85%83%2F%E5%8F%B0%2C%E6%B1%9F%E8%8B%8F%E8%AF%BA%E5%8A%9B%E6%9C%BA%E6%A2%B0%E8%AE%BE%E5%A4%87%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8%2C%E6%B1%9F%E8%8B%8F%E5%8D%97%E4%BA%AC&xzhid=31084720&category=%E4%BA%94%E9%87%91%E6%9C%BA%E7%94%B5%3B%E6%90%AC%E8%BF%90%E8%B5%B7%E9%87%8D%3B%E6%90%AC%E8%BF%90%E5%A0%86%E9%AB%98%E8%AE%BE%E5%A4%87&iid=e42d33fdab7f486d9b485af664fc697a&jid=550513976&prod_type=0"
    
    print(f"测试URL: {test_url}")
    print("=" * 80)
    
    # 测试地址提取
    full_address = crawler._get_full_address_from_detail(test_url)
    print(f"提取到的完整地址: {full_address}")
    
    if full_address:
        print("地址提取成功！")
    else:
        print("地址提取失败！")
    
    print("=" * 80)
    
    # 检查提取的地址是否包含详细信息
    if len(full_address) > 5:
        print(f"地址长度：{len(full_address)}，包含详细信息")
    else:
        print(f"地址长度：{len(full_address)}，可能只包含简单信息")

if __name__ == "__main__":
    test_address_extraction()
