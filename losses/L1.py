import torch.nn.functional as F

def masked_l1_loss(Original_img, pred_img, mask, weight_known=0.7, weight_missing=0.3):

    per_pixel_l1 = F.l1_loss(Original_img, pred_img, reduction='none')    #L1 loss
    pixel_weights = mask * weight_missing + (1 - mask) * weight_known   # 控制已知與未知的影響權重
    return (pixel_weights * per_pixel_l1).mean()    # 求取均值

def l1_loss(Original_img, pred_img):

    per_pixel_l1 = F.l1_loss(Original_img, pred_img, reduction='none')    # L1 loss
    return per_pixel_l1.mean()    # 求取均值

# def L1_loss_2():
#     loss = torch.nn.L1Loss()    #測試L1 loss 另一種寫法
#     return loss