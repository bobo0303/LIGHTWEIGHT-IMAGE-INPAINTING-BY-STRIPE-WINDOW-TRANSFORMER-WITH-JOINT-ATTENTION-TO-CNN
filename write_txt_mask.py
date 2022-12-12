import os, random
from random import shuffle
import cv2
import numpy as np

save_path = './txt/'

# # train_dataset
# train_path = 'C:/Users/Lab722-2080/image_inpainting/25_celeba/datasets/place2_256/data_256/'
# train_list = 'train_dataset_place2.txt'
#
#
# number = os.listdir(train_path)
#
# f = open(save_path+train_list, 'w')
# for n in number:
#     f.write(train_path + n + '\n')
# f.close()
#
# # mask_dataset
# mask_path = 'F:/mask_dataset/strokes/'
# range_list = [10, 15, 25, 25, 15, 10]
# mask_list = 'train_mask_place2.txt'
# mask_range_path = os.listdir(mask_path)
# fd = open(save_path+mask_list, 'w')
# number = len(os.listdir(train_path))    #len(train image) = len(mask image)
# mm=0
#
# for m in mask_range_path:
#     picknumber = int(number * float(range_list[mm]/100))
#     print(picknumber)
#     pathDir = os.listdir(mask_path+m)  # 取圖片的原始路徑
#     mm+=1
#     sample = random.sample(pathDir, picknumber)  # 隨機選取picknumber數量的樣本圖片
#     for n in sample:
#         fd.write(mask_path + m +'/' + n + '\n')
# fd.close()
#
# # test_path
# test_path = 'C:/Users/Lab722-2080/image_inpainting/25_celeba/datasets/place2_256/test_256_png/'
# test_list = 'test_dataset_place2.txt'
#
# number = os.listdir(test_path)
#
# f = open(save_path+test_list, 'w')
# for n in number:
#     f.write(test_path + n + '\n')
# f.close()

# 1~60 mask
mask_path = r'C:\Users\wiwiw\image_inpainting\datasets\CelebA\val_face'
file = os.listdir(mask_path)

mask_list = 'CelebA_strockes_val_10000_1_60.txt'
f = open(save_path+mask_list, 'w')
for nn in file:
    f.write(mask_path + nn+ '\n')
f.close()


