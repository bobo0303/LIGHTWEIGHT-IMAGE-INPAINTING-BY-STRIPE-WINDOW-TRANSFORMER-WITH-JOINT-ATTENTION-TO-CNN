import os
import cv2



frompath = 'E:/Place2/mask_dataset/10+20/' #這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
topath = 'E:/Place2/mask_dataset/color_changed/10+20/'

a = 'mask_256_00000574.bmp'
img = cv2.imread(frompath+a, cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
dst = 255 - img
newname=topath+'Places2_test00000574_mask.png' #在本案例中的命名規則為：年份+ - + 次序，最後一個.wav表示該檔案的型別
cv2.imwrite(newname, dst)
