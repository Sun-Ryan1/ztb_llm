import os
import re
import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin

class DataCollection:
    """数据收集模块
"""文本预处理：去除标点、特殊符号、多余空格"""if not text:
            return ""
        # 保留中文、英文、数字，去除其他符号
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        # 去除多余空格/换行
        text = re.sub(r'\s+', '', text)
        return text

    def _get_random_headers(self):
"""生成随机请求头，用于反爬"""return {
            'User-Agent': random.choice(self.user_agents),
            'Referer': random.choice(self.referers),  # 添加随机Referer
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',  # 兼容http->https跳转
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'DNT': '1',  # Do Not Track请求头，减少被追踪
            'X-Forwarded-For': f'{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}',
            'X-Requested-With': random.choice(['XMLHttpRequest', ''])
        }

    def _send_request_with_retry(self, url, timeout=20):
"""带重试机制的HTTP请求发送方法"""max_retries = 5
        
        for attempt in range(max_retries):
            try:
                # 使用随机请求头
                headers = self._get_random_headers()
                response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                response.encoding = response.apparent_encoding

                if response.status_code == 200:
                    return response
                else:
                    print(f"尝试{attempt+1}/{max_retries}：请求{url}失败，状态码: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"尝试{attempt+1}/{max_retries}：请求{url}失败，错误: {e}")
            
            if attempt < max_retries
"""爬取单个页面的政策文件"""response = self._send_request_with_retry(url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')

        # 只提取公告列表中的链接（针对所有网址的通用处理）
        # 尝试查找各种可能的公告列表容器
        announcement_containers = [
            soup.find('ul', class_='article-list'),  # 通用文章列表
            soup.find('div', class_='news-list'),  # 新闻列表
            soup.find('div', class_='announcement-list'),  # 公告列表
            soup.find('div', class_='tender-list'),  # 招标列表
            soup.find('div', class_='bid-list'),  # 中标列表
            soup.find('div', class_='procurement-list'),  # 采购列表
            soup.find('div', class_='article-content'),  # 文章内容区
            soup.find('div', class_='main-content')  # 主内容区
        ]
        
        # 找到第一个有效的公告列表容器
        announcement_list = None
        for container in announcement_containers:
            if container:
                announcement_list = container
                break
        
        if announcement_list:
            # 只从公告列表中提取链接
            links = [a for a in announcement_list.find_all('a') 
                    if a.get('href') and a.text.strip()]
            print(f"从页面找到{len(links)}个公告链接")
        else:
            # 回退到默认规则，但严格过滤非公告链接
            base_keywords = ['公告', '招标', '中标', '采购', '公示', '通知']
            links = [a for a in soup.select('a') 
                    if a.get('href') and len(a.get('href'))>5 and 
                    any(kw in a.text.strip() or kw in a.get('title', '') for kw in base_keywords)]
            print(f"从页面找到{len(links)}个有效公告链接（回退规则）")

        # 按数据源适配关键词（核心+细分场景）
        if source_name == 'faigui':
            # 法规专门关键词集
            base_keywords = [
                '法律', '法规', '条例', '办法', '规定', '细则', '准则', '标准',
                '通知', '意见', '指导意见', '管理办法', '实施办法', '暂行办法',
                '发布', '实施', '修订', '解读', '废止', '印发', '转发',
                '政府采购', '招标投标', '采购法', '招投标法', '公共资源交易'
            ]
            tender_keywords = base_keywords + ['财政部', '发改委', '住建部', '工信部', '水利部', '交通部',
                                              '国务院', '省政府', '市政府', '规范性文件', '政策文件']
        else:
            # 非法规数据源的关键词
            base_keywords = [
                '招标', '投标', '采购', '政府采购', '竞标', '中选', '成交', '中标',
                '通知', '办法', '规定', '公告', '公示', '细则', '方案', '意见', '指导意见', '管理办法',
                '公共资源', '交易', '印发', '发布', '实施', '修订', '解读', '暂行'
            ]
            tender_keywords = base_keywords + ['财政', '发改委', '住建', '工信', '水利', '交通']
            
        collected_files = []
        # 遍历找到的链接，逐个爬取文件内容
        for link in links:
            # 从文本或title属性获取标题
            file_title = link.text.strip() or link.get('title', '').strip()
            if not file_title:
                continue

            # 完善URL拼接逻辑（处理相对路径）
            file_url = link.get('href')
            if not file_url:
                continue
            if not file_url.startswith(('http://', 'https://')):
                file_url = urljoin(url, file_url)

            # 过滤无效链接：JS伪链接、无效域名
            if file_url.startswith('javascript:'):
                continue
            invalid_domains = ['rz.moe.gov.cn', 'jp.moe.gov.cn', 'en.moe.gov.cn']
            if any(domain in file_url for domain in invalid_domains):
                continue

            # 去重：跳过已爬取的URL
            if file_url in self.crawled_urls:
                continue
            self.crawled_urls.add(file_url)

            # 预处理标题（去标点/空格）
            clean_title = self.clean_text(file_title)
            print(f"检查标题：{file_title}（预处理后：{clean_title}）")

            # 获取文件内容（用于标题+内容双维度匹配）
            file_content = self.get_file_content(file_url)
            clean_content = self.clean_text(file_content) if file_content else ""

            # 改进的双维度匹配：标题或内容含关键词，同时支持部分匹配和容错
            title_match = any(kw in clean_title for kw in tender_keywords)
            content_match = any(kw in clean_content for kw in tender_keywords)
            # 增加容错：如果标题或内容较长，且包含部分关键词相关词汇，也认为匹配
            partial_match = False
            if len(clean_title) > 15 or len(clean_content) > 40:
                # 只需要匹配少数几个关键词即可，降低匹配门槛
                matched_keywords = [kw for kw in tender_keywords if kw in clean_title or kw in clean_content]
                # 根据标题/内容长度动态调整匹配阈值
                if len(clean_title) > 50 or len(clean_content) > 200:
                    partial_match = len(matched_keywords) >= 1  # 长内容只需1个关键词
                else:
                    partial_match = len(matched_keywords) >= 1  # 降低要求到1个关键词
            
            # 增加紧急容错：对于来源可信的网站，适当放宽匹配条件
            trusted_domains = ['gov.cn', 'gdzwfw.gov.cn', 'cebpubservice.com', 'ccgp.gov.cn']
            domain_match = any(domain in file_url for domain in trusted_domains)
            
            # 组合匹配策略：精确匹配 OR 部分匹配 OR 可信域名匹配
            final_match = title_match or content_match or partial_match or domain_match

            # 保存有效文件（内容非空且长度达标，使用新的组合匹配策略）
            if ((final_match) and file_content and len(file_content) > 25):  # 降低内容长度要求
                try:
                    # 生成唯一文件名（时间戳+序号）
                    file_name = f'{source_name}_{int(time.time())}_{len(collected_files)+1}.txt'
                    file_path = os.path.join(self.output_dir, source_name, file_name)
                    
                    # 保存文件（含完整元信息）
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"标题: {file_title}\n")
                        f.write(f"来源: {source_name}\n")
                        f.write(f"URL: {file_url}\n")
                        f.write(f"爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"内容: {file_content[:5000]}\n")  # 限制内容长度，避免文件过大
                    
                    collected_files.append(file_path)
                    print(f"✅ 成功收集: {file_title}")

                    # 随机延迟（1-3秒），降低反爬风险
                    time.sleep(random.uniform(1, 3))
                except Exception as e:
                    print(f"❌ 保存{file_url}失败: {e}")
            else:
                print(f"❌ 未命中关键词：{file_title}")

        return collected_files

    def crawl_website(self, url, source_name, start_page=1):
"""爬取指定网站的政策文件"""
        # 根据数据源类型和URL类型确定爬取策略
        if source_name == 'faigui':
            print(f"检测到法律法规数据源，只爬取当前页面")
            return self.crawl_single_page(url, source_name)
        
        # 处理中国政府采购网的招标/中标分页
        if 'ccgp.gov.cn/bxsearch' in url:
            collected_files = []
            print(f"检测到中国政府采购网招标/中标公告，开始爬取{start_page}-40页数据")
            
            # 爬取start_page-40页
            for page in range(start_page, 41):
                # 使用正则表达式动态替换page_index参数
                page_url = re.sub(r'page_index=(\d+)', f'page_index={page}', url)
                
                print(f"\n--
"""获取网页内容，支持多种编码自动识别
"""
        try:
            # 使用重试机制发送请求
            response = self._send_request_with_retry(url, timeout=15)
            if not response:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尝试找到主要内容区域
            main_content = None
            content_selectors = [
                '.content', '.main-content', '.article-content', '.article',
                '#content', '#main-content', '#article-content', '#article'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                # 如果找不到特定的内容区域，尝试排除导航栏、侧边栏等
                for tag in soup.find_all(['nav', 'aside', 'footer', 'header']):
                    tag.decompose()
                main_content = soup.find('body')
            
            if main_content:
                # 只提取文本内容，去除所有HTML标签
                text = main_content.get_text(separator='\n', strip=True)
                return text
            else:
                return ""
        except Exception as e:
            print(f"获取{url}内容失败: {e}")
            return ""

    def crawl_all_sources(self):
        """爬取所有数据源的政策文件
"""
        all_collected_files = []
        
        for source_name, urls in self.data_sources.items():
            print(f"\n")
            print(f"开始爬取数据源: {source_name}")
            print(f"")
            
            for url in urls:
                try:
                    print(f"\n正在爬取: {url}")
                    collected_files = self.crawl_website(url, source_name)
                    all_collected_files.extend(collected_files)
                    
                    # 不同数据源之间增加较长延迟，降低被识别为爬虫的风险
                    time.sleep(random.uniform(5, 10))
                except Exception as e:
                    print(f"爬取{url}失败: {e}")
                    continue
        
        return all_collected_files

    def crawl_only_faigui(self):
        """专门爬取法规数据源的政策文件
"""
        faigui_collected_files = []
        
        if 'faigui' in self.data_sources:
            print(f"\n")
            print(f"开始专门爬取法规数据源")
            print(f"")
            
            urls = self.data_sources['faigui']
            for url in urls:
                try:
                    print(f"\n正在爬取法规网站: {url}")
                    collected_files = self.crawl_website(url, 'faigui')
                    faigui_collected_files.extend(collected_files)
                    
                    # 不同法规网站之间增加较长延迟
                    time.sleep(random.uniform(5, 10))
                except Exception as e:
                    print(f"爬取法规网站{url}失败: {e}")
                    continue
        else:
            print("未找到法规数据源配置")
        
        return faigui_collected_files

    def crawl_faigui_from_url(self, url):
        """从指定URL爬取法规数据
"""
        print(f"\n")
        print(f"开始从指定URL爬取法规数据: {url}")
        print(f"")
        
        try:
            collected_files = self.crawl_website(url, 'faigui')
            print(f"\n爬取完成，共收集到{len(collected_files)}个法规文件")
            return collected_files
        except Exception as e:
            print(f"爬取指定URL失败: {e}")
            return []

