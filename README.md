# CSWin Transformer with joint attention for image inpainting (ICME 2023)

Reproduction of Nvidia image inpainting paper "Image Inpainting for Irregular Holes Using Partial Convolutions" https://arxiv.org/abs/1804.07723

- mdoel structure
<img src="https://i.imgur.com/wUjHZ2e.png" alt="https://i.imgur.com/wUjHZ2e.png" title="https://i.imgur.com/wUjHZ2e.png" width="5248" height="1888">

10,000 iteration results  (completion, output, mask)  
<img src="https://imgur.com/KxYDgF3.jpg"  width="768" height="512">

100,000 iteration results  (completion, output, mask)  
<img src="https://imgur.com/7pQiyAx.jpg"  width="768" height="512">

# Environment
- Python 3.7.11
- chainer  6.7.0   
- opencv (only for cv.imread, you can replace it with PIL)  
- PIL  

or see the requirements.txt

# How to try

## Download dataset (place2、CelebA)
[Place2](http://places2.csail.mit.edu/)  
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  

## Set dataset path

Edit common/paths.py
```python
train_place2 = "/yourpath/place2/data_256/"
val_place2 = "/yourpath/place2/val_256/"
test_place2 = "/yourpath/test_256/"

```
## Preprocessing  
In this implementation, masks are automatically generated in advance.  
```python
python generate_windows.py image_size generate_num
```
"image_size" indicates image size of masks.  
"generate_num" indicates the number of masks to generate.  

Default implementation uses image_size=256 and generate_num=1000.  
Note that original paper uses 512x512 image and generate mask with different way. 

## Run training
```python
python train.py -g 0 
```
-g represents gpu option.(utilize gpu of No.0) 

## Result

- Place2

<img src="https://imgur.com/P2XHsSA.jpg" width="700" height="600">

- CelebA 

<img src="https://imgur.com/ARzFFZv.jpg" width="700" height="600">

- COMPLEXITY COMPARISON

<img src="https://imgur.com/hjZB4k7.jpg" width="861" height="131">

All training and testing base on same 2080 Ti.

## Visual comparisons
- Place2

<img src="https://imgur.com/ysnqyN1.jpg" width="600" style="zoom:100%;">

- CelebA

<img src="https://imgur.com/JF8mzU0.jpg" width="600" style="zoom:100%;">

## Difference from original paper
Firstly, check [implementation FAQ](http://masc.cs.gmu.edu/wiki/partialconv)
1. C(0)=0 in first implementation (already fix in latest version)
2. Masks are generated using random walk by generate_window.py
3. To use chainer VGG pre-traied model, I re-scaled input of the model. See updater.vgg_extract. It includes cropping, so styleloss in outside of crop box is ignored.)
4. Padding is to make scale of height and width input:output=2:1 in encoder stage.  

other differences:  
- image_size=256x256 (original: 512x512)


## Acknowledgement
This repository utilizes the codes of following impressive repositories   
- [chainer-cyclegan](https://github.com/Aixile/chainer-cyclegan)  
- [partialconv](https://github.com/NVIDIA/partialconv)
- [chainer-partial_convolution_image_inpainting](https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting)

---
## Contact
If you have any question, feel free to contact wiwi61666166@gmail.com
