import torch.nn.functional as F

def Edge_MSE_loss(Original_img, pred_img, Edge):

    loss = F.mse_loss(Original_img, pred_img, reduction='none')
    loss = loss * (1 - Edge) + loss * Edge * 10
    return loss.mean()

