import json
import os

def calculate_average_accuracy(file_path):
    """计算指定JSON文件中所有测试用例的平均准确率
"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: {file_path} is not a JSON array")
            return None
        
        total_accuracy = 0.0
        total_cases = len(data)
        
        if total_cases == 0:
            print(f"Warning: {file_path} has no test cases")
            return 0.0
        
        for case in data:
            if 'accuracy' in case:
                total_accuracy += case['accuracy']
            else:
                print(f"Warning: Test case {case.get('test_case_id', 'unknown')} has no accuracy field")
        
        average_accuracy = total_accuracy / total_cases
        return average_accuracy, total_cases
    
    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: {file_path} not found")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # 定义文件路径
    scenario1_file = 'qwen_3B_bge-m3_scenario1_results.json'
    scenario2_file = 'qwen_3B_bge-m3_scenario2_results.json'
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整文件路径
    scenario1_path = os.path.join(current_dir, scenario1_file)
    scenario2_path = os.path.join(current_dir, scenario2_file)
    
    # 计算准确率
    print("=== 准确率计算结果 ===")
    
    # 处理scenario1
    result1 = calculate_average_accuracy(scenario1_path)
    if result1:
        avg_acc1, total1 = result1
        print(f"Scenario 1 ({scenario1_file}):")
        print(f"  测试用例总数: {total1}")
        print(f"  平均准确率: {avg_acc1:.4f} ({avg_acc1*100:.2f}%)")
    
    print()
    
    # 处理scenario2
    result2 = calculate_average_accuracy(scenario2_path)
    if result2:
        avg_acc2, total2 = result2
        print(f"Scenario 2 ({scenario2_file}):")
        print(f"  测试用例总数: {total2}")
        print(f"  平均准确率: {avg_acc2:.4f} ({avg_acc2*100:.2f}%)")
    
    print("====================")

if __name__ == "__main__":
    main()