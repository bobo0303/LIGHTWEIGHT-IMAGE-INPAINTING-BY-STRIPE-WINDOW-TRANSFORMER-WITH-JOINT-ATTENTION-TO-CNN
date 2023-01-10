# LIGHTWEIGHT IMAGE INPAINTING BY STRIPE WINDOW TRANSFORMER WITH JOINT ATTENTION TO CNN

You can find our paper ["Here"](https://arxiv.org/abs/2301.00553)

The overview of our proposed model. The whole model structure is the main model of our proposed and show the detail of our joint attention between Global layer and Local layer. The input feature only Im and M, the Iedge won’t be trained in the model and be generated by Canny before training. Moreover, the right side show the CSWin Transformer Block. At last, the Residual Dense Block in local layer show at the top right corner of the whole model.
<img src="https://i.imgur.com/PWw3CpW.png" alt="https://i.imgur.com/PWw3CpW.png" title="https://i.imgur.com/PWw3CpW.png" width="1312" height="350">

# Environment
- Python 3.7.0
- pytorch
- opencv
- PIL  
- colorama

or see the requirements.txt

# How to try

## Download dataset (places2、CelebA、ImageNet)
[Places2](http://places2.csail.mit.edu/)  
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
[ImageNet](https://www.image-net.org/download.php)

## Set dataset path

Edit txt/xxx.txt (set path in config)
```python

data_path = './txt/train_path.txt'
mask_path = './txt/train_mask_path.txt'
val_path = './txt/val_path.txt'
val_mask_path = './val_mask_file/' # path
test_path: './txt/test_path.txt'
test_mask_1_60_path: './test_mask_1+10_file/' # path

```

txt example
```python

E:/Places2/data_256/00000001.jpg
E:/Places2/data_256/00000002.jpg
E:/Places2/data_256/00000003.jpg
E:/Places2/data_256/00000004.jpg
E:/Places2/data_256/00000005.jpg
 ⋮

```

## Preprocessing  
In this implementation, masks are automatically generated by ourself. stroke masks mixed randomly to generate proportion from 1% to 60%.

strokes (from left to right 20%-30% 30%-40% 40%-50% 50%-60%)
<img src="https://imgur.com/m3CStkN.png" alt="https://imgur.com/m3CStkN.png" title="https://imgur.com/m3CStkN.png" width="1000" height="200">

## Pretrained model
["Here"](https://drive.google.com/drive/u/4/folders/1QeLZc7_4TZVZ7awRQXNmgvGtPl6OGUAR)

## Run training
```python

python train.py (main setting data_path/mask_path/val_path/val_mask_path/batch_size/train_epoch)

```
1. set the config path ('./config/model_config.yml')
2. Set path and parameter details in model_config.yml

Note: If the training is interrupted and you need to resume training, you can set resume_ckpt and resume_D_ckpt.

## Run testing
```python

python test.py (main setting test_ckpt/test_path/test_mask_1_60_path/save_img_path)

```
1. set the config path ('./config/model_config.yml')
2. Set path and parameter details in model_config.yml

## Quantitative comparison

- Places2

<img src="https://imgur.com/94ToBGm.jpg" width="1312" height="350">

- CelebA 

<img src="https://imgur.com/aQW3dp7.jpg" width="1312" height="350">

Quantitative evaluation of inpainting on Places2 dataset. We report Peak signal-to-noise ratio (PSNR) and structural similarity (SSIM) metrics. The ▲ denotes larger, and ▼ denotes lesser of the parameters compared to our proposed model. (Bold means the 1st best; Underline means the 2nd best; Italics means the 3rd best)

- LPIPS (Places2)

<div align=center>
<img src="https://imgur.com/GoJL936.jpg" width="550" height="150">
</div>

All training and testing base on same 3060.

## Qualitative comparisons

- Places2

<img src="https://imgur.com/MXPw6V5.jpg" width="1000" style="zoom:100%;">

Qualitative results of Places2 dataset among all compared models. From left to right: Masked image, RW, DeepFill_v2, HiFill, Iconv, AOT-GAN, CRFill, TFill, and Ours. Zoom-in for details.

- CelebA

<img src="https://imgur.com/kpLYqj3.jpg" width="1000" style="zoom:100%;">

Qualitative results of Places2 dataset among all compared models. From left to right: Masked image, RW, DeepFill_v2, Iconv, AOT-GAN, CRFill, TFill, and Ours. Zoom-in for details.

## Ablation study

- Transformer and HSV loss

<div align=center>
<img src="https://imgur.com/cY1bKyp.jpg" width="410" height="150"><img src="https://imgur.com/Utxgfzs.jpg" width="410" height="150">
</div>

(left) : Ablation study label of transformer and HSV experiment.

(right) : Ablation study of color deviation on inpainted images. From left to right: Masked images, w/o TotalHSV loss, and TotalHSV loss (w/o V).

## Object removal

<div align=center>
<img src="https://imgur.com/YegPtfj.jpg" width="1300" height="350">
</div>

Object removal (size 256×256) results. From left to right: Original image, mask, object removal result.


## Acknowledgement
This repository utilizes the codes of following impressive repositories   
- [ZITS](https://github.com/DQiaole/ZITS_inpainting)
- [LaMa](https://github.com/saic-mdal/lama)
- [CSWin Transformer](https://github.com/microsoft/CSWin-Transformer)
- [Vision Transformer](https://github.com/google-research/vision_transformer)

---
## Contact
If you have any question, feel free to contact wiwi61666166@gmail.com

## Cite
```

@article{liu2023lightweight,
  title={Lightweight Image Inpainting by Stripe Window Transformer with Joint Attention to CNN},
  author={Liu, Tsung-Jung and Chen, Po-Wei and Liu, Kuan-Hsien},
  journal={arXiv preprint arXiv:2301.00553},
  year={2023}
}

```

