import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 加载CSV文件
csv_path = '/home/yuanqw/Wlx/event-rgb-aligned/align_data_simple.csv'
df = pd.read_csv(csv_path)

# 原始npz文件所在的文件夹路径
npz_folder_path = '/home/yuanqw/Wlx/SAI_data/simple_raw'

# 新的npz文件将要存放的文件夹路径
new_npz_folder_path = '/home/yuanqw/Wlx/SAI_data/simple_raw_align'

# 如果新文件夹不存在，创建它
if not os.path.exists(new_npz_folder_path):
    os.makedirs(new_npz_folder_path)

# 使用tqdm包装循环以显示进度条
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    i = row['i']
    time_diff = row['time_diff']
    
    # 构建原始npz文件的完整路径
    npz_file_path = os.path.join(npz_folder_path, f'{i:04d}.npz')
    
    # 构建新npz文件的完整路径
    new_npz_file_path = os.path.join(new_npz_folder_path, f'{i:04d}.npz')
    
    # 检查原始文件是否存在
    if os.path.exists(npz_file_path):
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        
        # 修改events数组的第三列
        ts = data['occ_free_aps_ts']
        ts = ts + time_diff
        
        # 准备要保存的数据，除了events之外还包括原始数据中的其他变量
        save_data = {key: data[key] for key in data.files if key != 'occ_free_aps_ts'}
        save_data['occ_free_aps_ts'] = ts        
        # 保存修改后的events数组以及其他变量到新的npz文件中
        np.savez(new_npz_file_path, **save_data)
    else:
        print(f"文件 {npz_file_path} 不存在。")
