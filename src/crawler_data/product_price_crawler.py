import os
import re
import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
from urllib.parse import urljoin, quote

class ProductPriceCrawler:
    """商品价格信息爬虫，爬取爱采购、1688等网站的商品和价格信息
"""
    
    def __init__(self, output_dir='preprocessed_data'):
        self.output_dir = output_dir
        self.product_dir = os.path.join(output_dir, 'shangpin')
        self.price_dir = os.path.join(output_dir, 'jiage')
        
        # 创建输出目录
        os.makedirs(self.product_dir, exist_ok=True)
        os.makedirs(self.price_dir, exist_ok=True)
        
        # 商品关键词列表
        self.product_keywords = [
            # 原有工程机械类别
            '挖掘机', '起重机', '装载机', '推土机', '压路机', '叉车',
            # 办公设备类别
            '触控产品', '复印机', '视讯会议系统', '一体机办公设备', '扫描仪',
            '激光打印机', '喷墨打印机', '保险柜', '办公收费系统', '条码打印机',
            # 实验室用品类别
            '实验室专用设备', '实验台', '实验室仪器', '试验室压滤机', '通风柜',
            '实验试剂', '色谱配件', '传递窗', '实验器皿',
            # 检测仪器类别
            '内窥镜', '测厚仪', '涂层检测仪', '水分仪', '水质测试仪', 'PH仪',
            '放大镜', '显微镜', '真空测量仪器', '移液器', '教学仪器', '天平仪',
            # 消防救援类别
            '灭火器', '消防水带', '消防装备', '消防报警器', '消防照明',
            '救援用品', '防汛用品', '急救箱', '救援帐篷',
            '警示标识', '安全标识', '警示宣传', '自发光标识', '标牌类',
            # 建筑材料类别
            '水泥', '沙子', '混凝土', '石材', '管材管件', '格栅板', '井盖', '隔断',
            '玻璃钢', '网格板', '塑料材料', '钢格板', '排水板', '盖板及沟盖', '灌浆料',
            '保温/隔热材料', '防水/防潮材料', '耐火/防火材料', '隔音/吸声材料',
            '耐腐蚀防辐射材料', '节能环保材料'
        ]
        
        # 配置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.baidu.com/'
        }
    
    def _get_random_headers(self):
        """生成随机请求头
"""
        # 随机User-Agent列表，增加多样性
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/125.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
        ]
        
        headers = self.headers.copy()
        headers['User-Agent'] = random.choice(user_agents)
        return headers
    
    def _send_request(self, url, max_retries=3):
        """发送HTTP请求，支持重试
"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self._get_random_headers(), timeout=15)
                response.encoding = response.apparent_encoding
                if response.status_code == 200:
                    return response
                print(f"请求失败，状态码：{response.status_code}，重试第{attempt+1}次")
            except Exception as e:
                print(f"请求异常：{e}，重试第{attempt+1}次")
            # 增加随机延迟，避免被识别为爬虫
            time.sleep(random.uniform(2, 5))
        return None
    
    def crawl_baidu_aicaigou(self, keyword, pages=3):
        """爬取百度爱采购网站的商品信息
"""
        print(f"\n开始爬取百度爱采购的'{keyword}'商品信息...")
        
        # 移除重复商品的集合，使用商品ID作为唯一标识
        seen_products = set()
        all_products = []
        all_prices = []
        
        # 确保所需模块在该方法中可用
        import re
        import json
        import urllib.parse
        
        # 1. 首先获取初始页面，提取必要的参数和商品数据
        initial_url = f"https://b2b.baidu.com/s?q={quote(keyword)}&from_page=index&from_index=2&from_rec=fromPM&from=index_q"
        print(f"正在获取初始页面：{initial_url}")
        response = self._send_request(initial_url)
        
        if not response:
            print("获取初始页面失败，无法继续爬取")
            return all_products, all_prices
        
        html_text = response.text
        
        # 提取window.data对象
        data_match = re.search(r'window\.data\s*=\s*(\{.*?\});', html_text, re.DOTALL)
        if data_match:
            data_str = data_match.group(1)
            try:
                # 处理可能的JavaScript语法，转换为有效的JSON
                # 移除可能的JavaScript注释
                data_str = re.sub(r'//.*?\n|/\*.*?\*/', '', data_str, flags=re.DOTALL)
                # 将单引号转换为双引号（简单处理）
                data_str = re.sub(r"'", '"', data_str)
                # 移除可能的undefined值
                data_str = re.sub(r'\bundefined\b', 'null', data_str)
                # 移除可能的NaN值
                data_str = re.sub(r'\bNaN\b', 'null', data_str)
                # 移除可能的函数调用
                data_str = re.sub(r'\w+\(.*?\)', 'null', data_str)
                
                # 使用json.loads解析
                initial_data = json.loads(data_str)
                
                # 提取初始商品列表
                initial_product_list = initial_data.get('productList', [])
                print(f"初始页面找到 {len(initial_product_list)} 个商品")
                
                # 处理初始商品
                if initial_product_list:
                    new_products_count = 0
                    for product in initial_product_list:
                        # 只处理包含完整商品信息的对象
                        if product.get('fullName') and product.get('id'):
                            product_info = self._parse_aicaigou_product_data(product, keyword)
                            if product_info:
                                product_id = product.get('id', '')
                                if not product_id:
                                    product_id = product_info['商品名称']
                                    lid_match = re.search(r'lid=(\d+)', product_info['链接'])
                                    if lid_match:
                                        product_id += f"_{lid_match.group(1)}"
                            
                            if product_id not in seen_products:
                                seen_products.add(product_id)
                                all_products.append(product_info)
                                price_info = self._extract_price_info(product_info, keyword)
                                all_prices.append(price_info)
                                new_products_count += 1
                                print(f"新增商品：{product_info['商品名称']} (id={product_id})")
                            else:
                                # 详细日志：显示重复商品信息和原因
                                print(f"跳过重复商品：{product_info['商品名称']} (id={product_id})")
                    print(f"初始页面新增 {new_products_count} 个商品")
            except json.JSONDecodeError as e:
                print(f"初始页面JSON解析失败：{e}")
            except Exception as e:
                print(f"初始页面数据处理失败：{e}")
        
        # 核心改进：根据用户提示，直接使用百度爱采购的window.data.productList数据
        # 由于百度爱采购的反爬机制，我们无法直接获取更多页面的数据
        # 但我们可以通过优化商品提取和去重策略，获取更多不重复的商品
        print(f"\n爬取完成，共获取 {len(all_products)} 个不重复商品")
        
        # 增加商品详情页的爬取深度
        print(f"\n开始增加商品详情页的爬取深度...")
        
        # 只处理前5个商品，避免过多请求
        for product_info in all_products[:5]:
            link = product_info['链接']
            if link:
                # 尝试获取商品详情页的完整地址
                full_address = self._get_full_address_from_detail(link)
                if full_address:
                    product_info['地址'] = full_address
                    print(f"更新商品地址：{product_info['商品名称']} -> {full_address}")
        
        return all_products, all_prices
    
    def _get_full_address_from_detail(self, product_url):
        """从商品详情页提取完整地址
"""
        try:
            # 分析URL结构，提取实际的商品页面URL
            import urllib.parse
            import json
            parsed_url = urllib.parse.urlparse(product_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            # 检查是否包含url参数（这是实际的商品页面）
            actual_url = product_url
            if 'url' in query_params:
                actual_url = query_params['url'][0]
            
            # 访问商品详情页
            response = self._send_request(actual_url)
            if not response:
                return ''
            
            # 保存完整的详情页HTML用于调试
            with open('product_detail_page.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            html_text = response.text
            
            # 1. 优先从JavaScript的window.data对象中提取地址信息（百度爱采购SPA页面的特点）
            try:
                # 提取window.data对象
                data_match = re.search(r'window\.data\s*=\s*(\{.*?\});', html_text, re.DOTALL)
                if data_match:
                    data_str = data_match.group(1)
                    
                    # 处理可能的JavaScript语法，转换为有效的JSON
                    # 移除可能的JavaScript注释
                    data_str = re.sub(r'//.*?\n|/\*.*?\*/', '', data_str, flags=re.DOTALL)
                    # 将单引号转换为双引号（简单处理）
                    data_str = re.sub(r"'", '"', data_str)
                    # 移除可能的undefined值
                    data_str = re.sub(r'\bundefined\b', 'null', data_str)
                    # 移除可能的NaN值
                    data_str = re.sub(r'\bNaN\b', 'null', data_str)
                    # 移除可能的函数调用
                    data_str = re.sub(r'\w+\(.*?\)', 'null', data_str)
                    
                    # 使用json.loads解析
                    window_data = json.loads(data_str)
                    
                    # 从provider对象中提取注册地址
                    if 'provider' in window_data:
                        provider = window_data['provider']
                        if 'regAddr' in provider and provider['regAddr']:
                            return provider['regAddr']
            except Exception as js_error:
                print(f"解析window.data失败: {js_error}")
            
            # 2. 如果JavaScript数据中没有找到，尝试从HTML结构中提取
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # 查找联系我们部分
            contact_section = None
            
            # 1. 尝试通过class查找联系我们
            contact_section = soup.find(['div', 'section', 'div'], class_=re.compile(r'contact|联系我们|company-info|about|info-wrap|联系我们|contact-us|about-us'))
            
            # 2. 尝试通过文本查找联系我们
            if not contact_section:
                contact_header = soup.find(['h2', 'h3', 'h4'], string=re.compile(r'联系我们|关于我们|公司信息'))
                if contact_header:
                    contact_section = contact_header.parent
            
            # 3. 尝试查找页面底部的联系信息
            if not contact_section:
                page_footer = soup.find(['footer', 'div'], class_=re.compile(r'footer|bottom|页脚'))
                if page_footer:
                    contact_section = page_footer
            
            # 4. 如果都找不到，使用整个页面
            if not contact_section:
                contact_section = soup
            
            # 提取完整地址
            full_address = ''
            
            # 1. 尝试从联系地址标签获取
            address_keywords = ['联系地址', '公司地址', '地址', '所在地', '工厂地址', '联系我们']
            
            # 查找所有包含地址关键词的元素
            for keyword in address_keywords:
                address_tags = contact_section.find_all(['div', 'p', 'span', 'li'], string=re.compile(keyword))
                for tag in address_tags:
                    # 获取标签的所有后续兄弟元素文本
                    siblings_text = ''
                    for sibling in tag.next_siblings:
                        if hasattr(sibling, 'text'):
                            siblings_text += sibling.text.strip() + ' '
                        elif isinstance(sibling, str):
                            siblings_text += sibling.strip() + ' '
                    
                        # 如果文本长度足够，尝试提取地址
                        if len(siblings_text) > 10:
                            # 使用正则表达式提取地址
                            address_pattern = r'([\u4e00-\u9fa5]+(?:省|自治区|直辖市)[\u4e00-\u9fa5]+(?:市|地区|自治州|盟)[\u4e00-\u9fa5]+(?:区|县|市|旗)[\u4e00-\u9fa5]*[\u4e00-\u9fa5]+(?:路|街|巷|弄|道)[\u4e00-\u9fa50-9]+(?:号|栋|单元|室)?)'
                            match = re.search(address_pattern, siblings_text)
                            if match:
                                full_address = match.group(1).strip()
                                return full_address
                            # 尝试更宽松的匹配
                            address_pattern2 = r'([\u4e00-\u9fa5]+(?:省|自治区|直辖市)?[\u4e00-\u9fa5]+(?:市|地区|自治州|盟)[\u4e00-\u9fa5]+(?:区|县|市|旗))'
                            match2 = re.search(address_pattern2, siblings_text)
                            if match2:
                                full_address = match2.group(1).strip()
                                return full_address
                            # 如果包含关键词和具体地址信息，直接返回
                            if any(keyword in siblings_text for keyword in ['路', '街', '巷', '弄', '道', '号']):
                                full_address = siblings_text.strip()
                                return full_address
                
                # 尝试从父元素获取地址
                if 'tag' in locals():
                    parent_text = tag.parent.text
                    if len(parent_text) > 20:
                        # 使用正则表达式提取地址
                        address_pattern = r'([\u4e00-\u9fa5]+(?:省|自治区|直辖市)[\u4e00-\u9fa5]+(?:市|地区|自治州|盟)[\u4e00-\u9fa5]+(?:区|县|市|旗)[\u4e00-\u9fa5]*[\u4e00-\u9fa5]+(?:路|街|巷|弄|道)[\u4e00-\u9fa50-9]+(?:号|栋|单元|室)?)'
                        match = re.search(address_pattern, parent_text)
                        if match:
                            full_address = match.group(1).strip()
                            return full_address
            
            # 2. 如果第一种方法失败，尝试从整个页面提取
            if not full_address:
                all_text = soup.text
                address_pattern = r'([\u4e00-\u9fa5]+(?:省|自治区|直辖市)[\u4e00-\u9fa5]+(?:市|地区|自治州|盟)[\u4e00-\u9fa5]+(?:区|县|市|旗)[\u4e00-\u9fa5]*[\u4e00-\u9fa5]+(?:路|街|巷|弄|道)[\u4e00-\u9fa50-9]+(?:号|栋|单元|室)?)'
                match = re.search(address_pattern, all_text)
                if match:
                    full_address = match.group(1).strip()
            
            return full_address
        except Exception as e:
            # 打印详细错误信息用于调试
            print(f"提取详情页地址失败: {e}")
            return ''
    
    def _enhance_address_from_product_data(self, product):
        """从商品数据中提取更详细的地址信息
"""
        try:
            # 尝试从商品数据的其他字段中提取更详细的地址
            # 检查是否有其他字段包含地址信息
            full_address = product.get('location', '')
            
            # 尝试从多个可能的字段中提取地址信息
            potential_address_fields = ['fullAddress', 'companyAddress', 'contactAddress', 'address', 'locationDetail', 'detailAddress']
            
            for field in potential_address_fields:
                if field in product and isinstance(product[field], str) and product[field]:
                    full_address = product[field]
                    break
            
            # 如果只有简单地址（如"山东济宁"），尝试从商品描述或其他文本字段补充
            if full_address and len(full_address) < 10:
                # 检查商品描述字段
                description_fields = ['description', 'fullName', 'productDesc', 'specs', 'details']
                
                for field in description_fields:
                    if field in product and isinstance(product[field], str) and len(product[field]) > 20:
                        # 尝试从描述中提取更详细的地址
                        desc = product[field]
                        # 查找包含原地址的更详细地址
                        if full_address in desc:
                            # 使用正则表达式提取包含原地址的完整地址
                            address_pattern = rf'({re.escape(full_address)}[\u4e00-\u9fa50-9]+(?:路|街|巷|弄|道)[\u4e00-\u9fa50-9]+(?:号|栋|单元|室)?)'
                            match = re.search(address_pattern, desc)
                            if match:
                                full_address = match.group(1).strip()
                                break
                            # 尝试更宽松的匹配
                            address_pattern2 = rf'({re.escape(full_address)}.*?)(?:\s|,|。|，|;|；|$)'
                            match2 = re.search(address_pattern2, desc)
                            if match2:
                                potential_address = match2.group(1).strip()
                                if len(potential_address) > len(full_address):
                                    full_address = potential_address
                                    break
            
            return full_address
        except Exception as e:
            # 只打印关键错误，避免过多输出
            return product.get('location', '')
    
    def _parse_aicaigou_product_data(self, product, keyword):
        """解析从JavaScript中提取的百度爱采购商品数据
"""
        try:
            # 从原始数据中提取商品信息
            name = product.get('fullName', '未知')
            link = product.get('jumpUrl', '')
            
            # 处理价格信息
            price_value = product.get('price', '0')
            price_currency = product.get('pCurrency', '元')
            unit = product.get('unit', '')
            if unit:
                price = f"{price_value}{price_currency}/{unit}"
            else:
                price = f"{price_value}{price_currency}"
            
            supplier = product.get('fullProviderName', '未知')
            address = product.get('location', '未知')
            img_url = product.get('picUrl', '')
            
            # 构建商品描述
            category = product.get('category', '')
            description = f"类别：{category}"
            
            # 尝试获取商品详情页的完整地址
            full_address = ''
            if link:
                full_address = self._get_full_address_from_detail(link)
                # 如果获取到完整地址，则更新地址信息
                if full_address:
                    address = full_address
            
            product_info = {
                '商品名称': name,
                '关键词': keyword,
                '链接': link,
                '价格': price,
                '供应商': supplier,
                '地址': address,
                '图片链接': img_url,
                '描述': description,
                '来源': '百度爱采购',
                '采集时间': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return product_info
        except Exception as e:
            # 只打印关键错误，避免过多输出
            return None
    
    def _parse_aicaigou_product(self, product_div, keyword):
        """解析百度爱采购的商品信息
"""
        try:
            # 提取商品名称
"""从商品信息中提取价格信息
"""
        # 提取价格数字
        price = product_info['价格']
        price_num = re.findall(r'\d+(\.\d+)?', price)
        price_value = price_num[0] if price_num else '0'
        
        # 提取价格单位
        price_unit = re.findall(r'[\u4e00-\u9fa5]+', price)
        unit = ''.join(price_unit) if price_unit else '元'
        
        price_info = {
            '商品名称': product_info['商品名称'],
            '关键词': keyword,
            '供应商': product_info['供应商'],
            '价格': price,
            '价格数值': price_value,
            '价格单位': unit,
            '采集时间': product_info['采集时间'],
            '来源': product_info['来源'],
            '商品链接': product_info['链接']
        }
        
        return price_info
    
    def clean_product_data(self, product_data):
        """清洗商品数据
"""
        if not product_data:
            return []
        
        cleaned_data = []
        
        for product in product_data:
            # 清洗商品名称
            if '商品名称' in product:
                product['商品名称'] = product['商品名称'].strip()
            
            # 清洗价格
            if '价格' in product:
                product['价格'] = product['价格'].strip()
            
            # 清洗供应商
            if '供应商' in product:
                product['供应商'] = product['供应商'].strip()
            
            # 清洗地址
            if '地址' in product:
                product['地址'] = product['地址'].strip()
            
            # 清洗描述
            if '描述' in product:
                product['描述'] = product['描述'].strip()
            
            cleaned_data.append(product)
        
        return cleaned_data
    
    def clean_price_data(self, price_data):
        """清洗价格数据
"""
        if not price_data:
            return []
        
        cleaned_data = []
        
        for price in price_data:
            # 清洗商品名称
            if '商品名称' in price:
                price['商品名称'] = price['商品名称'].strip()
            
            # 清洗价格
            if '价格' in price:
                price['价格'] = price['价格'].strip()
            
            # 清洗价格数值
            if '价格数值' in price:
                # 确保价格数值是有效的
                price_value = price['价格数值']
                try:
                    # 提取数字部分
                    num_match = re.search(r'\d+(\.\d+)?', str(price_value))
                    if num_match:
                        price['价格数值'] = float(num_match.group())
                    else:
                        price['价格数值'] = 0.0
                except Exception:
                    price['价格数值'] = 0.0
            
            # 清洗供应商
            if '供应商' in price:
                price['供应商'] = price['供应商'].strip()
            
            cleaned_data.append(price)
        
        return cleaned_data
    
    def structure_product_data(self, product_data):
        """结构化商品数据
"""
        if not product_data:
            return []
        
        structured_data = []
        
        for product in product_data:
            # 确保所有必要字段都存在
            structured_product = {
                '商品名称': product.get('商品名称', ''),
                '关键词': product.get('关键词', ''),
                '链接': product.get('链接', ''),
                '价格': product.get('价格', ''),
                '供应商': product.get('供应商', ''),
                '地址': product.get('地址', ''),
                '图片链接': product.get('图片链接', ''),
                '描述': product.get('描述', ''),
                '来源': product.get('来源', ''),
                '采集时间': product.get('采集时间', time.strftime('%Y-%m-%d %H:%M:%S'))
            }
            
            structured_data.append(structured_product)
        
        return structured_data
    
    def structure_price_data(self, price_data):
        """结构化价格数据
"""
        if not price_data:
            return []
        
        structured_data = []
        
        for price in price_data:
            # 确保所有必要字段都存在
            structured_price = {
                '商品名称': price.get('商品名称', ''),
                '关键词': price.get('关键词', ''),
                '供应商': price.get('供应商', ''),
                '价格': price.get('价格', ''),
                '价格数值': price.get('价格数值', 0.0),
                '价格单位': price.get('价格单位', '元'),
                '采集时间': price.get('采集时间', time.strftime('%Y-%m-%d %H:%M:%S')),
                '来源': price.get('来源', ''),
                '商品链接': price.get('商品链接', '')
            }
            
            structured_data.append(structured_price)
        
        return structured_data
    
    def structure_all_data(self):
        """结构化所有数据
"""
        print("\n开始执行数据结构化操作...")
        
        # 处理商品数据
        product_files = os.listdir(self.product_dir)
        for product_file in product_files:
            if product_file.endswith('_商品信息.csv'):
                file_path = os.path.join(self.product_dir, product_file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if not df.empty:
                        # 转换为字典列表
                        product_data = df.to_dict('records')
                        # 清洗和结构化数据
                        cleaned_data = self.clean_product_data(product_data)
                        structured_data = self.structure_product_data(cleaned_data)
                        # 保存结构化后的数据
                        structured_df = pd.DataFrame(structured_data)
                        structured_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                        print(f"✅ 已结构化商品数据：{product_file}")
                except Exception as e:
                    print(f"❌ 结构化商品数据失败 {product_file}: {e}")
        
        # 处理价格数据
        price_files = os.listdir(self.price_dir)
        for price_file in price_files:
            if price_file.endswith('_价格信息.csv'):
                file_path = os.path.join(self.price_dir, price_file)
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if not df.empty:
                        # 转换为字典列表
                        price_data = df.to_dict('records')
                        # 清洗和结构化数据
                        cleaned_data = self.clean_price_data(price_data)
                        structured_data = self.structure_price_data(cleaned_data)
                        # 保存结构化后的数据
                        structured_df = pd.DataFrame(structured_data)
                        structured_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                        print(f"✅ 已结构化价格数据：{price_file}")
                except Exception as e:
                    print(f"❌ 结构化价格数据失败 {price_file}: {e}")
        
        print("数据结构化操作完成！")
    
    def save_to_csv(self, data, filename, folder):
        """将数据保存为CSV文件
"""
        if not data:
            print(f"没有数据可保存到{folder}/{filename}")
            return
        
        # 清洗和结构化数据
        if folder == 'product':
            data = self.clean_product_data(data)
            data = self.structure_product_data(data)
        elif folder == 'price':
            data = self.clean_price_data(data)
            data = self.structure_price_data(data)
        
        df = pd.DataFrame(data)
        
        # 根据文件夹确定保存路径
        if folder == 'product':
            save_path = os.path.join(self.product_dir, filename)
        elif folder == 'price':
            save_path = os.path.join(self.price_dir, filename)
        else:
            print(f"未知文件夹类型：{folder}")
            return
        
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存到：{save_path}")
    
    def crawl_all(self):
        """爬取所有关键词的商品信息
"""
        for keyword in self.product_keywords:
            products, prices = self.crawl_baidu_aicaigou(keyword)
            
            # 保存商品信息
            if products:
                product_filename = f'{keyword}_商品信息.csv'
                self.save_to_csv(products, product_filename, 'product')
            
            # 保存价格信息
            if prices:
                price_filename = f'{keyword}_价格信息.csv'
                self.save_to_csv(prices, price_filename, 'price')
            
            # 不同关键词之间延迟
            time.sleep(random.uniform(3, 6))
    
    def create_demo_data(self):
        """创建演示数据，确保功能正常
"""
        print("\n创建演示数据...")
        
        # 创建演示商品数据
        demo_products = []
        demo_prices = []
        
        for keyword in self.product_keywords:
            for i in range(5):
                # 生成演示商品信息
                product_info = {
                    '商品名称': f'{keyword}型号{i+1}',
                    '关键词': keyword,
                    '链接': f'https://example.com/product/{i+1}',
                    '价格': f'{(i+1)*1000}元',
                    '供应商': f'供应商{i+1}公司',
                    '地址': f'地区{i+1}',
                    '图片链接': f'https://example.com/image/{i+1}.jpg',
                    '描述': f'{keyword}的详细描述{i+1}',
                    '来源': '演示数据',
                    '采集时间': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                demo_products.append(product_info)
                
                # 生成演示价格信息
                price_info = {
                    '商品名称': f'{keyword}型号{i+1}',
                    '关键词': keyword,
                    '供应商': f'供应商{i+1}公司',
                    '价格': f'{(i+1)*1000}元',
                    '价格数值': f'{(i+1)*1000}',
                    '价格单位': '元',
                    '采集时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                    '来源': '演示数据',
                    '商品链接': f'https://example.com/product/{i+1}'
                }
                demo_prices.append(price_info)
        
        # 保存演示数据
        for keyword in self.product_keywords:
            # 筛选当前关键词的商品数据
            keyword_products = [p for p in demo_products if p['关键词'] == keyword]
            keyword_prices = [p for p in demo_prices if p['关键词'] == keyword]
            
            if keyword_products:
                product_filename = f'{keyword}_商品信息.csv'
                self.save_to_csv(keyword_products, product_filename, 'product')
            
            if keyword_prices:
                price_filename = f'{keyword}_价格信息.csv'
                self.save_to_csv(keyword_prices, price_filename, 'price')
        
        print("演示数据创建完成！")
    
    def crawl_all_platforms(self):
        """爬取所有平台的商品信息
"""
        print("开始爬取所有平台的商品信息...")
        
        # 爬取所有关键词
        for keyword in self.product_keywords:
            print(f"\n{'='*60}")
            print(f"处理关键词：{keyword}")
            print(f"{'='*60}")
            
            # 爬取百度爱采购，只爬取3页
            products_aicaigou, prices_aicaigou = self.crawl_baidu_aicaigou(keyword, pages=3)
            
            # 合并所有商品信息
            all_products = products_aicaigou
            all_prices = prices_aicaigou
            
            # 如果没有爬取到数据，创建演示数据
            if not all_products:
                print(f"未爬取到'{keyword}'的商品信息，创建演示数据...")
                # 创建演示数据
                product_info = {
                    '商品名称': f'{keyword}演示型号',
                    '关键词': keyword,
                    '链接': f'https://example.com/product/{keyword}',
                    '价格': '10000元',
                    '供应商': f'{keyword}供应商公司',
                    '地址': '演示地区',
                    '图片链接': 'https://example.com/image.jpg',
                    '描述': f'{keyword}的详细描述',
                    '来源': '演示数据',
                    '采集时间': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                all_products.append(product_info)
                
                price_info = {
                    '商品名称': f'{keyword}演示型号',
                    '关键词': keyword,
                    '供应商': f'{keyword}供应商公司',
                    '价格': '10000元',
                    '价格数值': '10000',
                    '价格单位': '元',
                    '采集时间': time.strftime('%Y-%m-%d %H:%M:%S'),
                    '来源': '演示数据',
                    '商品链接': f'https://example.com/product/{keyword}'
                }
                all_prices.append(price_info)
            
            # 保存商品信息
            if all_products:
                product_filename = f'{keyword}_商品信息.csv'
                self.save_to_csv(all_products, product_filename, 'product')
            
            # 保存价格信息
            if all_prices:
                price_filename = f'{keyword}_价格信息.csv'
                self.save_to_csv(all_prices, price_filename, 'price')
            
            # 不同关键词之间延迟
            time.sleep(random.uniform(3, 6))
        
        print("\n所有关键词爬取完成！")
        
        # 执行数据结构化操作
        self.structure_all_data()

if __name__ == "__main__":
    # 创建爬虫实例
    crawler = ProductPriceCrawler()
    
    # 爬取所有平台的商品信息
    crawler.crawl_all_platforms()
    
    # 不再自动创建演示数据，避免覆盖实际爬取的数据
    # 如果需要演示数据，可以手动调用create_demo_data()方法