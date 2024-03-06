#! /home/lyc/anaconda3/envs/esai/bin/python
# coding=utf-8
import sys
sys.path.append("/home/lyc/ros/catkin_build_ws/devel/lib/python3/dist-packages")

import os
import cv2
import numpy as np
import torch
from collections import OrderedDict
import csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
 

class ESAI:
    def __init__(self,num:int,period=2.5,model_path=None):
        self.num = num
        self.complex_path = os.path.abspath(f'/home/yuanqw/Wlx/SAI_data/complex_raw/{num:04d}.npz')
        #self.simple_path = os.path.abspath(f'/home/yuanqw/Wlx/SAI_data/simple_raw/{num}.npz')
        self.width,self.height = 346,260
        #self.period = period
        # self.bad_pixels = np.load(os.path.join(os.path.dirname(__file__),'bad_pixels.npy'))
        # self.bag_path = os.path.join(os.path.dirname(__file__),'bag','events.bag')
        self.num_events = 0
        self.event_list = []
        self.pos_save_path = os.path.join(os.path.dirname(__file__),f'results/pos_{num}')
        self.gt_save_path = os.path.join(os.path.dirname(__file__),f'results/gt_{num}')
        os.makedirs(self.gt_save_path,exist_ok=True)
        os.makedirs(self.pos_save_path,exist_ok=True)
        #input('Press Enter to start:')
    
    def get_file_name(path,suffix):
        """
        This function is used to get file name with specific suffix
        
        Parameters:
            path: path of the parent directory
            suffix: specific suffix (in the form like '.png')
        """
        name_list=[]
        file_list = os.listdir(path)
        for i in file_list:
            if os.path.splitext(i)[1] == suffix:
                name_list.append(i)
        name_list.sort()
        return name_list

    def preprocess(self):
        # Read x,y,t,p from rosdata
        data = np.load(self.complex_path,allow_pickle=True)
        eventData = data.get('events')     # unfocused events
        occ_free_aps_ts = data.get('occ_free_aps_ts')  # timestamp at the reference camera pose, which is equivalent to the timestamp of occlusion-free APS
        # k = data.get('k') # parameter from camera intrinsic matrix
        # p = data.get('p')
        #img_size = data.get('size') # image size
        occ_free_aps = data.get('occ_free_aps') # occlusion-free aps
 #**********************************************************************************
 
    ## manual event refocusing ------
        event_x = eventData[:,0]
        event_y = eventData[:,1]
        event_t = eventData[:,2]
        event_p = eventData[:,3]
        self.num_events = len(event_x)
        slice_num = 100
        ref_t_begin = event_t[int(self.num_events*0.45)]
        ref_t_end = event_t[int(self.num_events*0.7)]
        ref_t_list = np.linspace(ref_t_begin, ref_t_end, slice_num)
        #ref_t_list = np.vectorize(lambda x: format(x, '.7f'))(ref_t_list) # .4f保留四位小鼠
        time_step = 30
        minT = event_t.min()
        maxT = event_t.max()
        event_t_ind = event_t - minT
        interval = (maxT - minT) / time_step
        # filter events
        res_x,res_y = 256,256 # roi
        Xrange = ((self.width-res_x)//2,(self.width-res_x)//2+res_x)
        #Yrange = ((self.height-res_y)//2,(self.height-res_y)//2+100)
        Yrange = (0,120)
        # convert events to event tensors
        pos = np.zeros((time_step, self.height, self.width))
        neg = np.zeros((time_step, self.height, self.width))
        T,H,W = pos.shape
        pos_now = pos.ravel()
        neg_now = neg.ravel()
        ind = (event_t_ind / interval).astype(int)
        ind[ind == T] -= 1
        pos_ind = event_p == 1
        neg_ind = event_p == 0
        max_ssim = 0
        big_time = 0
        occ_free_aps_new = occ_free_aps[12, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
        occ_free_aps_new = (occ_free_aps_new - occ_free_aps_new.min()) / (occ_free_aps_new.max() - occ_free_aps_new.min())
        cv2.imwrite(os.path.join(self.gt_save_path,'gt.png'),occ_free_aps_new*255)
        print('Done! Results are saved in', self.gt_save_path)
        
        d = 1.41 # 目标深度 background is 1.41 frontier is 0.7
        fx = 421.2654  # 相机内参
        v = 0.1775 # 导轨速度
        
        for i, ref_t in enumerate(ref_t_list):
            # Refocus
            # v = 0.1434 # 导轨速度
            dt = event_t - ref_t
            dx = dt * v * fx / d
            event_x_change = event_x + dx
            event_x_change = np.clip(event_x_change, 0, self.width-1)
            
            # Generate event_tensor

            event_x_new = event_x_change.astype(int)
            event_y_new = event_y.astype(int)

            pos = pos_now
            neg = neg_now
            
            np.add.at(pos, event_x_new[pos_ind] + event_y_new[pos_ind]*W + ind[pos_ind]*W*H, 1)
            np.add.at(neg, event_x_new[neg_ind] + event_y_new[neg_ind]*W + ind[neg_ind]*W*H, 1)
            
            pos_new = np.reshape(pos, (T,H,W))
            neg_new = np.reshape(neg, (T,H,W))
            
            pos_new = pos_new[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
            neg_new = neg_new[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
            
            sum_pos = np.sum(pos_new+neg_new,axis=0)
            #sum_pos/= np.max(sum_pos)
            #pos_img = cv2.undistort(sum_pos*255,k, p)
            
            sum_pos = (sum_pos - sum_pos.min()) / (sum_pos.max() - sum_pos.min())
            #print(((occ_free_aps - occ_free_aps.min())/3).shape)
            delta_t = ref_t-occ_free_aps_ts[12]
            #print(occ_free_aps.min())
            cv2.imwrite(os.path.join(self.pos_save_path, 'pos{}.png'.format(i)), sum_pos*255)
            #cv2.imwrite(os.path.join(self.save_path, 'pos_after_distort.png'), pos_img)
            
            score = ssim(sum_pos*255, occ_free_aps_new*255)
            #iff = (diff * 255).astype("uint8")
            
            print("No.{}--SSIM:{}||Delta_t:{}".format(i,score,delta_t))
            if score > max_ssim: 
                max_ssim = score
                t = ref_t
                index = i
                final_time_diff = delta_t 
                big_time = 0
            else:
                big_time += 1
            
            if big_time > 20: break
            
        if index == 0:
            return "caution! should try below 0.45!",max_ssim, final_time_diff,t
        if index == slice_num-1:
            return "caution! should try up 0.7!" ,max_ssim, final_time_diff,t         
                
        return index, max_ssim, final_time_diff,t


if __name__ == "__main__":
    filename = os.path.join(os.path.dirname(__file__),'align_data.csv')
    with open(filename, mode='w',newline='') as file:
        # 写入标题行
        writer = csv.writer(file)
        file.write('i,ind,time_diff,refocus_time\n')
    for i in range(250):
        esai= ESAI(i)
        #esai.record()
        ind, ssim_max, time_diff, refocus_t= esai.preprocess()
        print("for data{}||the best_one is No{}||SSIM is {}|| delta t is {} ||refocus t is{} ".format(i,ind,ssim_max,time_diff,refocus_t))
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i,ind, time_diff, refocus_t])
            #file.write(f'{},{},{},{}\n'.format(i,ind, time_diff, refocus_t))
        print(f'数据已写入 {filename}')