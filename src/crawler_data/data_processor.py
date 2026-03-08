import os
import pandas as pd
import glob

class DataProcessor:
    """数据处理器，用于整合和预处理数据
"""
    
    def __init__(self, input_dir, output_dir, data_type):
        """初始化处理器
        
        Args:
            _dir: ，包含所有数据CSV文件
            output_dir: ，用于保存处理后的数据
            data_type: ，用于区分价格和商品数据
"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.data_type = data_type
    
    def load_all_data(self):
        """加载所有数据
"""
        print(f"正在加载{self.input_dir}目录下的所有{self.data_type}数据文件...")
        
        # 获取所有CSV文件路径
        csv_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        
        if not csv_files:
            print(f"在{self.input_dir}目录下未找到任何CSV文件")
            return pd.DataFrame()
        
        print(f"找到{len(csv_files)}个CSV文件")
        
        # 加载所有CSV文件并合并
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                all_data.append(df)
                print(f"成功加载: {os.path.basename(csv_file)} (共{len(df)}行)")
            except Exception as e:
                print(f"加载文件{os.path.basename(csv_file)}失败: {e}")
        
        if not all_data:
            print("未成功加载任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n数据合并完成，共{len(combined_df)}行数据")
        
        return combined_df
    
    def preprocess_data(self, df):
        """数据预处理
        
        Args:
            : 原始数据DataFrame
            
        Returns:
"""
        print(f"\n开始{self.data_type}数据预处理...")
        
        if df.empty:
            return df
        
        # 创建一个副本，避免修改原始数据
        processed_df = df.copy()
        
        # 1. 清理数据
        # 去除重复行
        initial_count = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        if len(processed_df) < initial_count:
            print(f"移除了{initial_count
"""保存处理后的数据到CSV文件
"""
        if df.empty:
            print("没有数据可保存")
            return False
        
        # 使用固定的文件名，与现有文件保持一致
        if self.data_type == "价格":
            output_file = os.path.join(self.output_dir, "integrated_price_data_20260102.csv")
        elif self.data_type == "商品":
            output_file = os.path.join(self.output_dir, "integrated_product_data_20260102.csv")
        else:
            output_file = os.path.join(self.output_dir, f"integrated_{self.data_type}_data_20260102.csv")
        
        print(f"\n正在保存处理后的{self.data_type}数据到: {output_file}")
        
        try:
            # 保存为CSV文件，使用UTF-8编码以支持中文
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"{self.data_type}数据成功保存到: {output_file}")
            print(f"保存的数据行数: {len(df)}")
            return True
        except Exception as e:
            print(f"保存{self.data_type}数据失败: {e}")
            return False
    
    def run(self):
        """执行完整的数据处理流程
"""
        print("="*60)
        print(f"开始{self.data_type}数据处理")
        print("="*60)
        
        # 1. 加载所有数据
        raw_data = self.load_all_data()
        
        if raw_data.empty:
            print(f"没有{self.data_type}数据需要处理")
            return False
        
        # 2. 数据预处理
        processed_data = self.preprocess_data(raw_data)
        
        # 3. 保存处理后的数据
        success = self.save_processed_data(processed_data)
        
        print("="*60)
        print(f"{self.data_type}数据处理完成")
        print("="*60)
        
        return success

    def generate_summary(self):
        """生成数据摘要报告
"""
        print(f"\n" + "="*60)
        print(f"{self.data_type}数据摘要")
        print("="*60)
        
        # 查找固定命名的输出文件
        if self.data_type == "价格":
            output_file = os.path.join(self.output_dir, "integrated_price_data_20260102.csv")
        elif self.data_type == "商品":
            output_file = os.path.join(self.output_dir, "integrated_product_data_20260102.csv")
        else:
            output_file = os.path.join(self.output_dir, f"integrated_{self.data_type}_data_20260102.csv")
        
        if not os.path.exists(output_file):
            print(f"未找到{self.data_type}输出文件: {output_file}")
            return
        
        # 加载处理后的数据
        df = pd.read_csv(output_file, encoding='utf-8-sig')
        
        print(f"总数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        print(f"数据列名: {', '.join(df.columns)}")
        
        # 统计商品分类数量
        category_columns = [col for col in df.columns if any(keyword in col for keyword in ['关键词', '分类', '商品名称', '名称'])]
        if category_columns:
            category_stats = df[category_columns[0]].value_counts().sort_values(ascending=False)
            print(f"\n商品分类数量: {len(category_stats)}")
            print(f"\n各商品分类数据数量:")
            print(category_stats)
        
        print("="*60)

def main():
    """主函数
"""
    # 定义输入输出路径
    base_dir = r"e:\pycharmproject\TBProject\preprocessed_data"
    
    # 处理价格数据
    jiage_input_dir = os.path.join(base_dir, "jiage")
    jiage_output_dir = os.path.join(base_dir, "jiage")
    jiage_processor = DataProcessor(jiage_input_dir, jiage_output_dir, "价格")
    jiage_processor.run()
    jiage_processor.generate_summary()
    
    print("\n" + "="*80)
    print("="*80)
    print("\n")
    
    # 处理商品数据
    shangpin_input_dir = os.path.join(base_dir, "shangpin")
    shangpin_output_dir = os.path.join(base_dir, "shangpin")
    shangpin_processor = DataProcessor(shangpin_input_dir, shangpin_output_dir, "商品")
    shangpin_processor.run()
    shangpin_processor.generate_summary()

if __name__ == "__main__":
    main()
