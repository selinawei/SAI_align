# SAI_align

本repository用于来解决rosbag录制过程中导致的aps帧和event帧的时差~
如果你发现event在对应时间refocus的结果和gt存在像素偏移，请使用本代码~~~

本代码主要利用SSIM比较不同对齐时间的效果（即是否和gt对齐）
并取用最对齐的一个，并用aps_ts - 该对齐方式的aps时间差
这样就可以得到aps和event对齐的npz啦~

关于本代码：
  ali_final_optimize 用于生成对齐文件align_data.csv，内涵第i个数据集，第ind个对齐slice，时差diff_t，对齐时间refocus_t

  aligned.py 用于查看raw数据情况（适用于npz）

  t_transform.py用于对raw npz中的aps_ts进行更改，得到一个组新的npz数据集
  thatsall~
  hope it could help u~
  >v<
