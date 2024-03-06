# import os
# import argparse
# import torch
# import numpy as np
# from utils import get_file_name, filter_events_by_key


# ar_load = np.load('Example_data/Processed-do/0000.npy', allow_pickle=True).item()
# pos =ar_load.get('slice_pic')     # unfocused events
# # pos = torch.FloatTensor(np.expand_dims(pos,axis=1))
# #eventData = ar_load.get('events')     # unfocused events
# print(pos)
# #print(len(eventData['x']))

import numpy as np

# 替换为你的npz文件路径
npz_file_path = '/home/yuanqw/Wlx/SAI_data/complex_raw/0000.npz'

# 加载npz文件
with np.load(npz_file_path) as data:
    # 遍历所有的数组
    for item in data:
        # 获取数组的名字
        name = item
        # 获取数组
        array = data[item]
        # 打印数组的名字和形状
        print(f"Name: {name}, Shape: {array.shape}")
        # 打印数组的值
        if name == 'occ_free_aps_ts':
            formatted_arr = np.vectorize(lambda x: format(x, '.7f'))(array) # .4f保留四位小鼠
            print(f"Values: {formatted_arr}\n")
        if name == 'events':
            print(f"Values: {array[100000][2]}\n")
        if name == 'k':
            print(f"Values: {array}\n")
        if name == 'p':
            print(f"Values: {array}\n")
        if name == 'v':
            print(f"Values: {array}\n")
        if name == 'size':
            print(f"Values: {array}\n")            



#p.save('Example_data/Raw/0000_new.npy', new_data)
