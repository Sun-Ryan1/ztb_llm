#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试查询分类逻辑
"""
import re

class TestQueryClassification:
    """测试查询分类逻辑
"""
    
    def __init__(self):
        self.test_cases = [
            # 信用代码查询
            ("统一信用代码91320981MA1Y77PKX8对应的公司", "credit_code"),
            ("信用代码91310116MA1JBLHNXM对应的企业是什么", "credit_code"),
            ("查询信用代码91320722MA1Y754A9X的企业信息", "credit_code"),
            # 地址查询
            ("地址在广东省 东莞市 万 江 街道 万 道路 79 号 410 室附近的公司", "address"),
            ("地址在南京市 鼓楼区 中山北路 6 号附近的公司", "address"),
            ("台州浙江省 台州市 临海市 头门港 新区地区的企业有哪些", "address"),
            ("四平路周边的公司", "address"),
            # 经营范围查询
            ("做服装服饰 及 面辅料业务的公司", "business_scope"),
            ("主要业务为建筑 机械设备 租赁 。 ( 依法 须 经 批准 的 项目的公司", "business_scope"),
            ("经营电子设备销售的企业", "business_scope"),
            # 其他查询
            ("华为技术有限公司", "general"),
            ("张三", "general")
        ]
    
    def _is_credit_code(self, query_text: str) -> bool:
        """判断是否为信用代码查询
"""
        # 信用代码正则：18位字母数字，通常以91开头，匹配大小写
        credit_code_pattern = r'91[0-9a-zA-Z]{16}'
        # 同时检测查询中是否包含"统一信用代码"等关键词
        credit_code_keywords = ['统一社会信用代码', '统一信用代码', '信用代码', '社会信用代码', '纳税人识别号']
        
        has_credit_code = bool(re.search(credit_code_pattern, query_text, re.IGNORECASE))
        has_credit_keyword = any(keyword in query_text for keyword in credit_code_keywords)
        
        return has_credit_code or has_credit_keyword
    
    def _is_address_query(self, query_text: str) -> bool:
        """判断是否为地址查询
"""
        # 地址关键词
        address_keywords = ['地址', '附近', '位于', '在', '注册地', '所在地', '坐落', '位置', '地址是', '位于', '周边', '地区', '区域']
        # 地名关键词
        location_keywords = ['市', '区', '县', '镇', '街道', '路', '巷', '弄', '号', '村', '乡', '大道', '街', '园区', '工业区', '新区']
        # 区域关键词
        region_keywords = ['中国', '省', '自治区', '直辖市', '特别行政区']
        
        # 检查是否包含地址相关关键词
        has_address_keyword = any(keyword in query_text for keyword in address_keywords)
        has_location_keyword = any(keyword in query_text for keyword in location_keywords)
        has_region_keyword = any(keyword in query_text for keyword in region_keywords)
        
        # 检查是否包含数字地址（如XX路XX号）
        has_numbered_address = bool(re.search(r'[路街道路巷弄号]+\d+', query_text))
        
        # 检查是否包含多个地名关键词（如XX市XX区）
        location_count = sum(1 for keyword in location_keywords if keyword in query_text)
        has_multiple_locations = location_count >= 2
        
        # 包含地址关键词，或者包含地名关键词+区域关键词，或者包含数字地址，或者包含多个地名关键词
        return has_address_keyword or (has_location_keyword and has_region_keyword) or has_numbered_address or has_multiple_locations
    
    def _is_business_scope_query(self, query_text: str) -> bool:
        """判断是否为经营范围查询
"""
        # 经营范围关键词
        business_keywords = [
            '经营范围', '主要业务', '从事', '业务', '经营', '项目', '许可',
            '业务范围', '经营项目', '许可项目', '一般项目', '主营业务',
            '经营内容', '生产范围', '服务范围', '销售范围', '经营范围包括',
            '做', '从事', '生产', '销售', '提供', '研发', '业务为', '主要为',
            '及', '的公司', '的企业'
        ]
        
        # 检查是否包含业务范围相关关键词
        has_business_keyword = any(keyword in query_text for keyword in business_keywords)
        
        # 检查是否包含行业相关关键词 + 经营/从事等动词
        industry_verbs = ['经营', '从事', '生产', '销售', '提供', '研发', '做']
        has_industry_verb = any(verb in query_text for verb in industry_verbs)
        
        # 检查是否包含典型的经营范围查询模式
        has_business_pattern = bool(re.search(r'[经营从事生产销售提供研发做].*[业务项目范围]', query_text)) or \
                              bool(re.search(r'[业务项目范围].*[经营从事生产销售提供研发做]', query_text)) or \
                              bool(re.search(r'做.*的公司', query_text)) or \
                              bool(re.search(r'经营.*的企业', query_text))
        
        return has_business_keyword or has_industry_verb or has_business_pattern
    
    def _classify_query(self, query: str) -> str:
        """简单分类查询类型用于统计
"""
        # 信用代码查询
        if self._is_credit_code(query):
            return "credit_code"
        # 地址查询
        elif self._is_address_query(query):
            return "address"
        # 经营范围查询
        elif self._is_business_scope_query(query):
            return "business_scope"
        # 其他查询
        else:
            return "general"
    
    def run_tests(self):
        """运行测试
"""
        print("=== 测试查询分类逻辑 ===")
        print(f"测试用例总数: {len(self.test_cases)}")
        
        passed = 0
        failed = 0
        
        for i, (query, expected_type) in enumerate(self.test_cases, 1):
            actual_type = self._classify_query(query)
            status = "✓" if actual_type == expected_type else "✗"
            
            if actual_type == expected_type:
                passed += 1
            else:
                failed += 1
            
            print(f"{i:2d}. {status} 查询: '{query}'")
            print(f"    预期类型: {expected_type}")
            print(f"    实际类型: {actual_type}")
            print()
        
        print("=== 测试结果 ===")
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"通过率: {passed / len(self.test_cases) * 100:.1f}%")
        
        return passed == len(self.test_cases)

if __name__ == "__main__":
    tester = TestQueryClassification()
    tester.run_tests()
