import os
import re
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    """数据预处理模块
"""

    def __init__(self, input_dir='enhanced_collected_data', output_dir='preprocessed_data'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def load_data_by_category(self, category):
        """加载指定分类的数据
"""
        data = []
        folder_path = os.path.join(self.input_dir, category)
        
        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return data
        
        print(f"正在加载{category}文件夹中的数据...")
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data.append(content)
                except Exception as e:
                    print(f"加载文件{file_path}失败: {e}")
        
        return data

    def load_data(self):
        """加载爬取的文本数据，支持分类后的文件夹结构
"""
        all_data = {}
        
        # 遍历三个子文件夹
        for folder in ['zhaobiao', 'zhongbiao', 'faigui']:
            all_data[folder] = self.load_data_by_category(folder)
        
        return all_data

    def _clean_text(self, text):
        """清理文本，保留指定的标点符号和字符
"""
        if not text:
            return ""
        # 保留指定的标点符号和字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.!?，。！？;:：；\s\\/\-\_\.\~\:\?\#\[\]\@\!\$\&\'\(\)\*\+\,\;\=]', '', text)
        # 标准化空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def preprocess_text(self, text):
        """预处理文本数据，完整保留文件结构和关键信息
"""
        if not text:
            return ""
        
        # 解析文件结构，保留完整的行结构
        lines = text.split('\n')
        processed_lines = []
        content_section = False
        
        for line in lines:
            if line.startswith('标题:'):
                # 保留标题行的格式，只对标题内容进行分词处理
                title_part = line[3:].strip()
                if title_part:
                    words = jieba.cut(title_part)
                    processed_title = ' '.join(words)
                    processed_lines.append(f'标题: {processed_title}')
                else:
                    processed_lines.append(line)
                content_section = False
            elif line.startswith('来源:') or line.startswith('爬取时间:'):
                # 保留来源和爬取时间行，不做分词处理
                processed_lines.append(line)
                content_section = False
            elif line.startswith('URL:'):
                # 完全保留URL行，不做任何处理
                processed_lines.append(line)
                content_section = False
            elif line.startswith('内容:'):
                # 内容部分开始
                processed_lines.append(line)
                content_section = True
            elif content_section:
                # 内容部分的每一行都单独处理，保留行结构
                if not line.strip():  # 保留空行
                    processed_lines.append('')
                    continue
                
                # 检查是否包含URL，如果包含则特殊处理
                original_line = line.strip()
                processed_line = original_line
                
                # 提取URL（更健壮的正则表达式，确保能匹配到完整的URL）
                url_pattern = r'(https?://[a-zA-Z0-9\-\._~:/\?#\[\]@!\$&\'\(\)\*\+,;=]+)'
                url_matches = list(re.finditer(url_pattern, original_line))
                urls = [match.group(0) for match in url_matches]
                
                if urls:
                    # 对每行进行分词，但保留URL的完整性
                    words = []
                    current_pos = 0
                    
                    for url in urls:
                        # 找到URL在原始行中的位置
                        pos = original_line.find(url, current_pos)
                        if pos != -1:
                            # 对URL之前的文本进行处理和分词
                            if pos > current_pos:
                                text_part = original_line[current_pos:pos]
                                # 处理文本部分，保留更多标点符号
                                text_part = self._clean_text(text_part)
                                if text_part:
                                    words.extend(jieba.cut(text_part))
                            # 直接添加完整的原始URL
                            words.append(url)
                            current_pos = pos + len(url)
                    
                    # 处理剩余文本
                    if current_pos < len(original_line):
                        text_part = original_line[current_pos:]
                        # 处理剩余文本，保留更多标点符号
                        text_part = self._clean_text(text_part)
                        if text_part:
                            words.extend(jieba.cut(text_part))
                    
                    if words:
                        processed_line = ' '.join(words)
                        processed_lines.append(processed_line)
                else:
                    # 不包含URL的情况，正常处理，保留更多标点符号
                    processed_line = self._clean_text(original_line)
                    if processed_line:
                        words = jieba.cut(processed_line)
                        processed_line = ' '.join(words)
                        processed_lines.append(processed_line)
            else:
                # 其他行直接保留
                processed_lines.append(line)
        
        # 使用换行符连接所有行，保持原始结构
        return '\n'.join(processed_lines)

    def process_data(self):
        """处理所有数据，分别保存到不同的文件夹中
"""
        all_data = self.load_data()
        
        all_results = {}
        
        for category, data in all_data.items():
            # 创建该分类的输出子文件夹
            category_output_dir = os.path.join(self.output_dir, category)
            if not os.path.exists(category_output_dir):
                os.makedirs(category_output_dir)
            
            preprocessed_data = []
            for i, text in enumerate(data):
                preprocessed = self.preprocess_text(text)
                preprocessed_data.append(preprocessed)
                
                # 保存预处理后的数据到分类子文件夹
                file_name = f'preprocessed_{category}_{i}.txt'
                file_path = os.path.join(category_output_dir, file_name)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(preprocessed)
            
            print(f"{category}: 已预处理{len(preprocessed_data)}个文件")
            
            # 保存到分类的 CSV 文件，放在分类子文件夹中
            category_csv_path = os.path.join(category_output_dir, f'{category}_preprocessed.csv')
            df = self.save_to_csv(preprocessed_data, category_csv_path)
            all_results[category] = df
        
        return all_results

    def vectorize_data(self, preprocessed_data):
        """将文本数据向量化
"""
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(preprocessed_data)
        
        return X, vectorizer

    def split_data(self, X, y=None, test_size=0.2, random_state=42):
        """划分训练集和测试集
"""
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            return X_train, X_test

    def preprocess_csv_file(self, csv_file_path):
        """预处理 CSV 文件中的文本内容
"""
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            print(f"正在处理 CSV 文件: {csv_file_path}")
            print(f"CSV 文件包含 {len(df)} 行数据")
            
            processed_data = []
            
            for index, row in df.iterrows():
                # 构建文本内容
                text_parts = []
                
                # 添加标题
                if '标题' in row and pd.notna(row['标题']):
                    text_parts.append(f"标题: {row['标题']}")
                
                # 添加来源
                if '来源' in row and pd.notna(row['来源']):
                    text_parts.append(f"来源: {row['来源']}")
                
                # 添加发布日期
                if '发布日期' in row and pd.notna(row['发布日期']):
                    text_parts.append(f"发布日期: {row['发布日期']}")
                
                # 添加原文链接
                if '原文链接' in row and pd.notna(row['原文链接']):
                    text_parts.append(f"URL: {row['原文链接']}")
                
                # 添加内容
                if '相关内容' in row and pd.notna(row['相关内容']):
                    text_parts.append(f"内容: {row['相关内容']}")
                
                # 合并所有部分
                full_text = '\n'.join(text_parts)
                
                # 预处理文本
                preprocessed = self.preprocess_text(full_text)
                processed_data.append(preprocessed)
            
            print(f"CSV 文件处理完成，共处理 {len(processed_data)} 条记录")
            return processed_data
            
        except Exception as e:
            print(f"处理 CSV 文件 {csv_file_path} 失败: {e}")
            return []

    def save_to_csv(self, preprocessed_data, csv_file_path='preprocessed_data.csv'):
        """将预处理后的数据保存为 CSV 文件
"""
        data_list = []
        
        for i, text in enumerate(preprocessed_data):
            data_dict = {}
            
            # 解析预处理后的文本
            lines = text.split('\n')
            
            content_section = False
            content_lines = []
            
            for line in lines:
                if line.startswith('标题:'):
                    data_dict['标题'] = line[3:].strip()
                    content_section = False
                elif line.startswith('来源:'):
                    data_dict['来源'] = line[3:].strip()
                    content_section = False
                elif line.startswith('爬取时间:'):
                    data_dict['爬取时间'] = line[5:].strip()
                    content_section = False
                elif line.startswith('URL:'):
                    data_dict['原文链接'] = line[4:].strip()
                    content_section = False
                elif line.startswith('内容:'):
                    # 内容部分开始，提取该行中的内容部分
                    content_start = line[3:].strip()
                    if content_start:
                        content_lines.append(content_start)
                    content_section = True
                elif content_section:
                    # 内容部分的每一行都添加到内容列表中
                    if line.strip():
                        content_lines.append(line.strip())
                else:
                    # 其他行，不处理
                    pass
            
            # 合并所有内容行
            if content_lines:
                data_dict['相关内容'] = ' '.join(content_lines)
            
            # 如果没有标题，使用索引作为标题
            if '标题' not in data_dict:
                data_dict['标题'] = f"文档_{i}"
            
            data_list.append(data_dict)
        
        # 创建 DataFrame 并保存
        df = pd.DataFrame(data_list)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(csv_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"预处理后的数据已保存至: {csv_file_path}")
        print(f"共保存 {len(df)} 条记录")
        
        return df

    def preprocess_enterprise_csv(self, csv_file_path, output_file_path=None):
        """预处理企业信息 CSV 文件，对文本字段进行清洗和分词
"""
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
            print(f"正在处理企业信息 CSV 文件: {csv_file_path}")
            print(f"CSV 文件包含 {len(df)} 行数据，列: {list(df.columns)}")
            
            # 定义需要预处理的文本字段
            text_fields = ['企业名称', '经营范围', '注册地址']
            
            # 对每行数据进行处理
            for index, row in df.iterrows():
                for field in text_fields:
                    if field in df.columns and pd.notna(row[field]):
                        # 清理文本
                        cleaned_text = self._clean_text(str(row[field]))
                        # 分词
                        words = jieba.cut(cleaned_text)
                        df.at[index, field] = ' '.join(words)
            
            # 确定输出文件路径
            if output_file_path is None:
                base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
                output_file_path = os.path.join(os.path.dirname(csv_file_path), f"{base_name}_processed.csv")
            
            # 保存处理后的数据
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"预处理后的数据已保存至: {output_file_path}")
            print(f"共保存 {len(df)} 条记录\n")
            
            return df
            
        except Exception as e:
            print(f"处理企业信息 CSV 文件 {csv_file_path} 失败: {e}")
            return None

    def preprocess_all_enterprise_csvs(self, qiye_dir='preprocessed_data/qiye'):
        """预处理 qiye 文件夹下的所有 CSV 文件
"""
        if not os.path.exists(qiye_dir):
            print(f"目录不存在: {qiye_dir}")
            return []
        
        csv_files = [f for f in os.listdir(qiye_dir) if f.endswith('.csv') and not f.endswith('_processed.csv')]
        
        if not csv_files:
            print(f"在 {qiye_dir} 目录中没有找到需要处理的 CSV 文件")
            return []
        
        print(f"找到 {len(csv_files)} 个需要处理的 CSV 文件\n")
        
        processed_dfs = []
        for csv_file in csv_files:
            csv_path = os.path.join(qiye_dir, csv_file)
            df = self.preprocess_enterprise_csv(csv_path)
            if df is not None:
                processed_dfs.append(df)
        
        return processed_dfs

    def preprocess_police_law_csv(self, csv_path):
        """预处理 police_law.csv 文件，创建新的预处理文件
"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            print(f"正在处理 police_law.csv: {csv_path}")
            print(f"CSV 文件包含 {len(df)} 行数据，列: {list(df.columns)}")
            
            # 对相关内容列进行预处理
            for index, row in df.iterrows():
                if '相关内容' in df.columns and pd.notna(row['相关内容']):
                    # 清理文本
                    cleaned_text = self._clean_text(str(row['相关内容']))
                    # 分词
                    words = jieba.cut(cleaned_text)
                    df.at[index, '相关内容'] = ' '.join(words)
            
            # 对标题列进行预处理
            for index, row in df.iterrows():
                if '标题' in df.columns and pd.notna(row['标题']):
                    # 清理文本
                    cleaned_text = self._clean_text(str(row['标题']))
                    # 分词
                    words = jieba.cut(cleaned_text)
                    df.at[index, '标题'] = ' '.join(words)
            
            # 保存到新的文件
            output_path = os.path.join(os.path.dirname(csv_path), 'police_law_processed.csv')
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"预处理后的数据已保存至: {output_path}")
            print(f"共保存 {len(df)} 条记录\n")
            
            return df
            
        except Exception as e:
            print(f"处理 police_law.csv 失败: {e}")
            return None

    def merge_preprocessed_txt_by_category(self, input_dir='preprocessed_data', category='zhaobiao', output_csv='preprocessed_data/zhaobiao_merged.csv'):
        """将指定分类的预处理 TXT 文件合并到一个 CSV 文件
"""
        # 定义 CSV 列
        columns = [
            '合同编号', '合同名称', '项目编号', '项目名称', '采购人(甲方)', '采购人地址', '采购人联系方式',
            '供应商(乙方)', '供应商地址', '供应商联系方式', '主要标的名称', '规格型号或服务要求',
            '主要标的数量', '主要标的单价', '合同金额(万元)', '履约期限、地点等简要信息', '采购方式',
            '合同签订日期', '合同公告日期', '其他补充事宜', '所属地域', '所属行业', '代理机构'
        ]
        
        # 获取指定分类的预处理 TXT 文件
        txt_files = [f for f in os.listdir(input_dir) if f.startswith(f'preprocessed_{category}_') and f.endswith('.txt')]
        
        if not txt_files:
            print(f"在 {input_dir} 目录中没有找到 {category} 分类的预处理 TXT 文件")
            return None
        
        print(f"找到 {len(txt_files)} 个 {category} 分类的预处理 TXT 文件")
        
        # 创建空的 DataFrame
        data_list = []
        
        for txt_file in txt_files:
            file_path = os.path.join(input_dir, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 解析文件内容
                data_dict = {col: '' for col in columns}
                
                # 提取标题
                title_match = re.search(r'标题:\s*(.+)', content)
                if title_match:
                    data_dict['合同名称'] = title_match.group(1).strip()
                
                # 提取来源
                source_match = re.search(r'来源:\s*(.+)', content)
                if source_match:
                    data_dict['所属行业'] = source_match.group(1).strip()
                
                # 提取 URL
                url_match = re.search(r'URL:\s*(.+)', content)
                if url_match:
                    data_dict['其他补充事宜'] = url_match.group(1).strip()
                
                # 提取爬取时间
                time_match = re.search(r'爬取时间:\s*(.+)', content)
                if time_match:
                    data_dict['合同公告日期'] = time_match.group(1).strip()
                
                # 提取内容
                content_match = re.search(r'内容:(.+)', content, re.DOTALL)
                if content_match:
                    data_dict['规格型号或服务要求'] = content_match.group(1).strip()
                
                data_list.append(data_dict)
                
            except Exception as e:
                print(f"处理文件 {txt_file} 失败: {e}")
        
        # 创建 DataFrame
        df = pd.DataFrame(data_list, columns=columns)
        
        # 保存到 CSV
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"{category} 合并后的数据已保存至: {output_csv}")
        print(f"共保存 {len(df)} 条记录\n")
        
        return df

if __name__ == "__main__":
    # 使用数据预处理模块处理数据并分别保存为 CSV
    print("开始预处理数据...")
    
    preprocessor = DataPreprocessing(input_dir='preprocessed_data', output_dir='preprocessed_data')
    
    # 处理所有数据，分别保存到不同的 CSV 文件
    print("=" * 60)
    print("开始处理招标、中标、法规数据...")
    print("=" * 60)
    
    all_results = preprocessor.process_data()
    
    print(f"\n")
    print(f"预处理完成！")
    for category, df in all_results.items():
        print(f"{category}: {len(df)} 条记录")
    print(f"\n")
    
    # 处理企业信息 CSV 文件
    print("=" * 60)
    print("开始处理企业信息 CSV 文件...")
    print("=" * 60)
    
    qiye_dir = 'e:\\pycharmproject\\TBProject\\preprocessed_data\\qiye'
    processed_dfs = preprocessor.preprocess_all_enterprise_csvs(qiye_dir=qiye_dir)
    
    print(f"\n")
    print(f"企业信息 CSV 文件处理完成！")
    print(f"共处理 {len(processed_dfs)} 个 CSV 文件")
    print(f"处理后的文件保存在: {qiye_dir}")
    print(f"\n")
    
    # 预处理 police_law.csv 文件
    print("=" * 60)
    print("开始处理 police_law.csv 文件...")
    print("=" * 60)
    
    police_law_csv = 'e:\\pycharmproject\\TBProject\\preprocessed_data\\faigui\\police_law.csv'
    preprocessor.preprocess_police_law_csv(police_law_csv)
    
    print(f"")
    print(f"police_law.csv 文件处理完成！")
    print(f"处理后的文件保存在: {os.path.join(os.path.dirname(police_law_csv), 'police_law_processed.csv')}")
    print(f"\n")
    
    # 合并预处理 TXT 文件到 CSV（按分类分别合并）
    print("=" * 60)
    print("开始合并预处理 TXT 文件到 CSV（按分类）...")
    print("=" * 60)
    
    for category in ['zhaobiao', 'zhongbiao', 'faigui']:
        category_dir = 'e:\\pycharmproject\\TBProject\\preprocessed_data'
        category_output = f'e:\\pycharmproject\\TBProject\\preprocessed_data\\{category}_merged.csv'
        
        # 获取该分类的预处理 TXT 文件
        txt_files = [f for f in os.listdir(category_dir) if f.startswith(f'preprocessed_{category}_') and f.endswith('.txt')]
        
        if txt_files:
            print(f"\n处理 {category} 分类，找到 {len(txt_files)} 个文件")
            merged_df = preprocessor.merge_preprocessed_txt_by_category(category_dir, category, category_output)
            
            if merged_df is not None:
                print(f"{category}: 合并完成，共 {len(merged_df)} 条记录")
    
    print(f"\n")
    print(f"所有预处理 TXT 文件合并完成！")
    print(f"")