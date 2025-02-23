import json
import os
from collections import Counter
from datetime import datetime
import pandas as pd
# [1]A little stressed, [2]Definitely stressed, [3]Stressed out, [4]Feeling good, [5]Feeling great
def process_stress_data(user_id):
    """处理用户的压力水平数据"""
    json_path = f'dataset/EMA/response/Stress/Stress_u{user_id}.json'
    output_dir = 'processed_stress'
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        stress_data = json.load(f)
    
    print(f"\n原始数据条数: {len(stress_data)}")
    
    # 打印原始数据样本
    print("\n原始数据样本:")
    for entry in stress_data[:5]:
        print(entry)
    
    # 打印时间范围
    all_times = [entry['resp_time'] for entry in stress_data if 'resp_time' in entry]
    if all_times:
        earliest = datetime.fromtimestamp(min(all_times))
        latest = datetime.fromtimestamp(max(all_times))
        print(f"\n数据时间范围: {earliest} 到 {latest}")
    
    # 按天组织数据
    daily_stress = {}
    valid_entries = 0
    invalid_entries = []
    
    for entry in stress_data:
        resp_time = entry.get('resp_time')
        if not resp_time:
            continue
            
        # 尝试从不同的键中获取压力值
        stress_level = None
        if 'level' in entry:
            stress_level = entry['level']
        elif 'null' in entry:
            stress_level = entry['null']
            
        if not stress_level:
            continue
            
        # 如果是 'Unknown'，跳过
        if stress_level == 'Unknown':
            invalid_entries.append((datetime.fromtimestamp(resp_time), stress_level, "Unknown值"))
            continue
            
        # 尝试将stress_level转换为整数，只接受1-5的值
        try:
            stress_level = int(float(stress_level))  # 允许字符串形式的数字
            if stress_level < 1 or stress_level > 5:
                invalid_entries.append((datetime.fromtimestamp(resp_time), stress_level, "超出范围"))
                continue
            
            # 将1-3归结为1，将4-5归结为0 (0 = 不压力，1 = 压力)
            if stress_level <= 3:
                stress_level = 1
            else:
                stress_level = 0
            # 记录有效的条目
            entry_time = datetime.fromtimestamp(resp_time)
            valid_entries += 1
            
            # 转换时间戳到日期
            date = entry_time.date()
            
            # 将stress level添加到对应的日期
            if date not in daily_stress:
                daily_stress[date] = []
            daily_stress[date].append(stress_level)
            
        except (ValueError, TypeError):
            # 如果包含逗号，可能是GPS坐标
            if isinstance(stress_level, str) and ',' in stress_level:
                invalid_entries.append((datetime.fromtimestamp(resp_time), stress_level, "GPS坐标"))
            else:
                invalid_entries.append((datetime.fromtimestamp(resp_time), stress_level, "无效格式"))
            continue
    
    print(f"\n有效压力值条目数: {valid_entries}")
    print("\n无效条目样本及原因:")
    for time, value, reason in invalid_entries[:5]:
        print(f"时间: {time}, 值: {value}, 原因: {reason}")
    
    # 计算每天最常见的压力水平
    daily_mode = {}
    for date, levels in sorted(daily_stress.items()):
        mode = Counter(levels).most_common(1)
        if mode:
            daily_mode[date] = mode[0][0]
            print(f"日期: {date}, 压力值: {mode[0][0]} (出现次数: {mode[0][1]}, 当天所有值: {levels})")
    
    # 将结果保存为CSV文件
    output_file = os.path.join(output_dir, f'stress_levels_u{user_id}.csv')
    df = pd.DataFrame(list(daily_mode.items()), columns=['date', 'stress_level'])
    df.sort_values('date', inplace=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n处理完成用户 {user_id} 的数据")
    print(f"总天数: {len(daily_mode)}")
    print(f"数据已保存到: {output_file}")
    
    return daily_mode

def process_all_users():
    """处理所有用户的数据"""
    for i in range(60):
        user_id = f"{i:02d}"
        try:
            print(f"\n处理用户 {user_id} 的数据...")
            process_stress_data(user_id)
        except Exception as e:
            print(f"处理用户 {user_id} 数据时出错: {str(e)}")
            print(f"错误类型: {type(e).__name__}")

if __name__ == "__main__":
    process_all_users() 