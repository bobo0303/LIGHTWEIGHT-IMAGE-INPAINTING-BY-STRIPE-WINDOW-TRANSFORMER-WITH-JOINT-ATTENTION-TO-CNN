import os
import random
from glob import glob
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from skimage.color import rgb2gray
from skimage.feature import canny
from torch.utils.data import Dataset

class get_dataset(Dataset):
    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None, is_train=False, image_size=256):

        self.is_train = is_train
        self.pt_dataset = pt_dataset

        self.image_id_list = []     #圖片list清空
        with open(self.pt_dataset) as f:    #開dataset資料夾 data_path OR validation_path
            for line in f:
                self.image_id_list.append(line.strip())     ##把裡面的圖片一個一個加進去

        if is_train:    #train的的部分 (train_dataset)
            self.strokes_mask_list = []   #清空
            with open(mask_path) as f:   #開第一個mask_path
                for line in f:
                    self.strokes_mask_list.append(line.strip())   #把裡面的圖片一個一個加進去
            self.strokes_mask_list = sorted(self.strokes_mask_list, key=lambda x: x.split('/')[-1])

            '''self.segment_mask_list = []   #清空
            with open(mask_path[1]) as f:   #開第二個mask_path
                for line in f:
                    self.segment_mask_list.append(line.strip())    #把裡面的圖片一個一個加進去
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])'''

        else:   #test的部分 (test_dataset)
            self.mask_list = glob(test_mask_path + '/*')    #test_mask_path 裡面所有圖片
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])      #重新排序 (應該是照數字順序排 以後面開始看的樣子)

        self.image_size = image_size    #default = 256
        self.training = is_train

    def __len__(self):
        return len(self.image_id_list)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]  #只取shape的 0,1 兩項 (高 寬)

        # test mode: load mask non random
        if self.training is False:  #test的部分
            mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)  #test 的mask 灰階讀進去
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)  #resize mask的部分
            mask = (mask > 127).astype(np.uint8) * 255  #mask大於127的*255 轉1 其他0
            return mask
        else:  # train mode:  random brush
            mask_index = random.randint(0, len(self.strokes_mask_list) - 1)   #隨機選某個
            mask = cv2.imread(self.strokes_mask_list[mask_index],
                              cv2.IMREAD_GRAYSCALE)  #讀隨機選出的那張圖片
            mask = (mask > 127).astype(np.uint8) * 255  #mask大於127的*255 轉1 其他0
            return mask

    def to_tensor(self, img, norm=False):
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   #歸1化 -1~1
        return img_t

    def load_edge(self, img):
        return canny(img, sigma=2, mask=None).astype(np.float)  #轉邊緣圖

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2] #只取shape的 0,1 兩項 (高 寬)

        if center_crop and imgh != imgw:    #如果center_crop(中心剪裁) AND 高寬不相同
            # center crop
            side = np.minimum(imgh, imgw)   #兩個取小的
            j = (imgh - side) // 2  #取整數
            i = (imgw - side) // 2  #取整數
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)     #resizze的部分
        return img

    def __getitem__(self, idx):
        selected_img_name = self.image_id_list[idx]
        img = cv2.imread(selected_img_name) #讀圖
        while img is None:
            print('Bad image {}...'.format(selected_img_name))  #讀不了圖 輸出 Bad image {選到的那張圖}
            idx = random.randint(0, len(self.image_id_list) - 1)    #隨機亂選費為內圖片
            img = cv2.imread(self.image_id_list[idx])   #讀圖
        img = img[:, :, ::-1]   #圖片的顏色通道為BGR，為了與原始圖片的RGB通道同步，需要轉換顏色通道

        img = self.resize(img, self.image_size, self.image_size, center_crop=False) #resize圖片
        img_gray = rgb2gray(img)    #讀成灰階
        edge = self.load_edge(img_gray) #轉 edge圖
        # load mask
        mask = self.load_mask(img, idx) #讀mask
        # augment data  擴充數據
        if self.training is True:   #只有 train_dataset會有
            if random.random() < 0.5:
                img = img[:, ::-1, ...].copy()  #翻轉Y
                edge = edge[:, ::-1].copy() #翻轉Y
            if random.random() < 0.5:
                mask = mask[:, ::-1, ...].copy()    #翻轉Y
            if random.random() < 0.5:
                mask = mask[::-1, :, ...].copy()    #翻轉X

        img = self.to_tensor(img, norm=False)
        edge = self.to_tensor(edge)
        mask = self.to_tensor(mask)
        #mask_img = img * (1 - mask)

        meta = {'img': img, 'mask': mask, 'edge': edge,
                'name': os.path.basename(selected_img_name)}
        return meta



