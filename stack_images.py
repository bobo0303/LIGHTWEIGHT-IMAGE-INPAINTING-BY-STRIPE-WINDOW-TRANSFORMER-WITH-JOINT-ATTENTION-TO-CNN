import numpy as np
import cv2

if __name__ =='__main__':
    img1 = cv2.imread(r'F:\ICME_2023\icme_Q\P1.png')
    img2 = cv2.imread(r'F:\ICME_2023\icme_Q\P2.png')
    imgs = np.vstack([img1, img2])
    # cv2.imshow('1',imgs)
    cv2.imwrite(r'F:\ICME_2023\icme_Q/P1_2.png', imgs)
