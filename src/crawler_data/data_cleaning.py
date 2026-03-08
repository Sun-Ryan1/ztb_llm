import pandas as pd
import numpy as np
import re
from datetime import datetime
import logging
import os

# 设置日志配置
logging.basicConfig(
    filename='cleaning_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 创建报告数据框
quality_report = pd.DataFrame(columns=[
    'file_name', 'total_records', 'missing_values', 'duplicate_records',
    'invalid_dates', 'cleaned_records', 'cleaning_time'
])

# 读取文件函数
def read_file(file_path):
    """读取CSV文件，处理编码问题
"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logging.info(f"成功读取文件: {file_path}")
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='gbk')
        logging.info(f"使用GBK编码读取文件: {file_path}")
        return df
    except Exception as e:
        logging.error(f"读取文件失败 {file_path}: {str(e)}")
        raise

# 日期标准化函数
def standardize_date(date_str):
    """将日期字符串转换为YYYY-MM-DD格式
"""
    if pd.isna(date_str) or date_str == '':
        return pd.NaT
    
    # 尝试多种日期格式
    formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日', '%Y-%m-%d %H:%M:%S',
               '%Y/%m/%d %H:%M:%S', '%Y年%m月%d日 %H:%M:%S']
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # 处理包含时间的日期字符串，如"2025年12月29日 18:17"
    try:
        date_part = str(date_str).split(' ')[0]
        for fmt in formats:
            if ' ' not in fmt:
                try:
                    return datetime.strptime(date_part, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
    except:
        pass
    
    logging.warning(f"无法解析日期: {date_str}")
    return pd.NaT

# 数值标准化函数
def standardize_numeric(num_str):
    """将数值字符串转换为标准数值格式
"""
    if pd.isna(num_str) or num_str == '' or str(num_str) == '面议':
        return np.nan
    
    num_str = str(num_str)
    
    # 去除千分位逗号
    num_str = re.sub(r',', '', num_str)
    
    # 提取数字部分
    num_parts = re.findall(r'[\d.]+', num_str)
    if not num_parts:
        logging.warning(f"无法提取数值: {num_str}")
        return np.nan
    
    num_part = num_parts[0]
    
    # 处理万元或万单位
    if '万元' in num_str or ('万' in num_str and len(num_parts) > 0):
        try:
            return float(num_part) * 10000
        except ValueError:
            logging.warning(f"无法转换万数值: {num_str}")
            return np.nan
    
    # 尝试转换其他数值
    try:
        return float(num_part)
    except ValueError:
        logging.warning(f"无法转换数值: {num_str}")
        return np.nan

# 缺失值处理函数
def handle_missing_values(df, core_fields, non_core_fields):
    """处理缺失值
"""
    # 记录初始记录数
    initial_count = len(df)
    
    # 核心字段缺失值处理：直接剔除
    df = df.dropna(subset=core_fields)
    
    # 非核心字段缺失值处理：填充"未知"或默认值
    for field in non_core_fields:
        if field in df.columns:
            df[field] = df[field].fillna('未知')
    
    # 记录处理后的记录数
    final_count = len(df)
    logging.info(f"缺失值处理：从{initial_count}条记录减少到{final_count}条记录")
    
    return df

# 重复值处理函数
def handle_duplicates(df, primary_key=None):
    """处理重复值
"""
    initial_count = len(df)
    
    if primary_key and primary_key in df.columns:
        # 基于主键去重
        df = df.drop_duplicates(subset=[primary_key], keep='first')
    else:
        # 基于所有列去重
        df = df.drop_duplicates(keep='first')
    
    final_count = len(df)
    logging.info(f"重复值处理：从{initial_count}条记录减少到{final_count}条记录")
    
    return df

# 文本净化函数
def clean_text(text):
    """净化文本内容
"""
    if pd.isna(text) or text == '':
        return '未知'
    
    text = str(text)
    # 去除多余的换行符和空格
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    return text.strip()

# 生成数据质量报告
def generate_quality_report(file_name, df_initial, df_cleaned):
    """生成数据质量报告
"""
    global quality_report
    
    # 计算缺失值数量
    missing_values = df_initial.isnull().sum().sum()
    
    # 计算重复值数量
    duplicate_count = len(df_initial)
"""清理company.csv文件
"""
    logging.info("开始清理company.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\company.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'company_name', 'credit_code']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    if 'registration_date' in df.columns:
        df['registration_date'] = df['registration_date'].apply(standardize_date)
    
    # 数值标准化
    if 'registered_capital_value' in df.columns:
        df['registered_capital_value'] = df['registered_capital_value'].apply(standardize_numeric)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['company_name', 'company_type', 'legal_representative', 
                  'province', 'city', 'registration_address', 'business_scope']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_company.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的company.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('company.csv', df_initial, df)
    
    return df

# 清理law.csv函数
def clean_law():
    """清理law.csv文件
"""
    logging.info("开始清理law.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\law.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'title', 'content']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    for date_field in ['publish_date', 'law_publish_date']:
        if date_field in df.columns:
            df[date_field] = df[date_field].apply(standardize_date)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['title', 'content', 'source', 'law_name', 'law_type', 'law_code']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_law.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的law.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('law.csv', df_initial, df)
    
    return df

# 清理product.csv函数
def clean_product():
    """清理product.csv文件
"""
    logging.info("开始清理product.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\product.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'product_name']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    for date_field in ['collection_time', 'collection_time_2']:
        if date_field in df.columns:
            df[date_field] = df[date_field].apply(standardize_date)
    
    # 数值标准化
    if 'price' in df.columns:
        df['price'] = df['price'].apply(standardize_numeric)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['product_name', 'keyword', 'supplier', 'address', 'description', 'product_category', 'supplier_2']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_product.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的product.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('product.csv', df_initial, df)
    
    return df

# 清理price.csv函数
def clean_price():
    """清理price.csv文件
"""
    logging.info("开始清理price.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\price.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'product_name']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    for date_field in ['collection_time', 'date']:
        if date_field in df.columns:
            df[date_field] = df[date_field].apply(standardize_date)
    
    # 数值标准化
    if 'price' in df.columns:
        df['price'] = df['price'].apply(standardize_numeric)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['product_name', 'keyword', 'supplier', 'source', 'product_category']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_price.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的price.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('price.csv', df_initial, df)
    
    return df

# 清理zhaobiao.csv函数
def clean_zhaobiao():
    """清理zhaobiao.csv文件
"""
    logging.info("开始清理zhaobiao.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\zhaobiao.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'contract_name', 'project_name']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    for date_field in ['contract_sign_date', 'contract_announce_date']:
        if date_field in df.columns:
            df[date_field] = df[date_field].apply(standardize_date)
    
    # 数值标准化
    if 'contract_amount' in df.columns:
        df['contract_amount'] = df['contract_amount'].apply(standardize_numeric)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['contract_name', 'project_name', 'buyer', 'buyer_address', 'buyer_contact',
                  'supplier', 'supplier_address', 'supplier_contact', 'main_target_name',
                  'specs', 'performance_info', 'procurement_method', 'other_supplements',
                  'administrative_region', 'industry', 'agency', 'source', 'purchase_type', 'project_type']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_zhaobiao.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的zhaobiao.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('zhaobiao.csv', df_initial, df)
    
    return df

# 清理zhongbiao.csv函数
def clean_zhongbiao():
    """清理zhongbiao.csv文件
"""
    logging.info("开始清理zhongbiao.csv")
    
    file_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\zhongbiao.csv'
    df = read_file(file_path)
    df_initial = df.copy()
    
    # 定义核心字段和非核心字段
    core_fields = ['id', 'contract_name', 'project_name']
    non_core_fields = [col for col in df.columns if col not in core_fields]
    
    # 日期标准化
    for date_field in ['contract_sign_date', 'contract_announce_date']:
        if date_field in df.columns:
            df[date_field] = df[date_field].apply(standardize_date)
    
    # 数值标准化
    if 'contract_amount' in df.columns:
        df['contract_amount'] = df['contract_amount'].apply(standardize_numeric)
    
    # 缺失值处理
    df = handle_missing_values(df, core_fields, non_core_fields)
    
    # 重复值处理
    df = handle_duplicates(df, 'id')
    
    # 文本净化
    text_fields = ['contract_name', 'project_name', 'buyer', 'buyer_address', 'buyer_contact',
                  'supplier', 'supplier_address', 'supplier_contact', 'main_target_name',
                  'specs', 'performance_info', 'procurement_method', 'other_supplements',
                  'administrative_region', 'industry', 'agency', 'source']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].apply(clean_text)
    
    # 保存清理后的数据
    output_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\cleaned_zhongbiao.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"清理后的zhongbiao.csv保存到: {output_path}")
    
    # 生成质量报告
    generate_quality_report('zhongbiao.csv', df_initial, df)
    
    return df

# 主函数
def main():
    """主执行函数
"""
    logging.info("开始数据清理任务")
    
    # 按顺序清理各个文件
    clean_company()
    clean_law()
    clean_product()
    clean_price()
    clean_zhaobiao()
    clean_zhongbiao()
    
    # 保存数据质量报告
    report_path = 'e:\\pycharmproject\\ZTB_Law_Product_Price_Company\\dataset\\data_quality_report.csv'
    quality_report.to_csv(report_path, index=False, encoding='utf-8')
    logging.info(f"数据质量报告保存到: {report_path}")
    
    logging.info("数据清理任务完成")

if __name__ == "__main__":
    main()
