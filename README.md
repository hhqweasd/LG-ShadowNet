# LG-ShadowNet
[Shadow Removal by a Lightness-Guided Network with Training on Unpaired Data.](https://ieeexplore.ieee.org/document/9318562)

```
@article{liu2021shadow,
  title={Shadow Removal by a Lightness-Guided Network with Training on Unpaired Data},
  author={Liu, Zhihao and Yin, Hui and Mi, Yang and Pu, Mengyang and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={1853--1865},
  year={2021},
  publisher={IEEE}
}
```

## Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.1+ & tochvision
- scikit-image

## Train and test on the adjusted ISTD dataset
Train 
1. Set the path of the dataset in ```train_aistd_module1.py```
2. Run ```train_aistd_module1.py```
3. Set the paths of the saved module1 models ```(netG_A2B.pth,netG_B2A.pth)``` and the dataset in ```train_aistd.py```
4. Run ```train_aistd.py```

Test   
1. Set the paths of the dataset and saved LG-ShadowNet models ```(netG_A2B.pth)``` in ```test_aistd.py```
2. Run ```test_aistd.py```

## Evaluate
1. Set the paths of the shadow removal result and the dataset in ```evaluate.m```
2. Run ```evaluate.m```

## Acknowledgments
Code is implemented based on [Mask-ShadowGAN](https://github.com/xw-hu/Mask-ShadowGAN).


## Results of LG-ShadowNet

GoogleDrive: [AISTD](https://drive.google.com/file/d/1psjWoflIK2tPC0mzMNyp-LYA1QkQeYtK/view?usp=sharing)/[ISTD](https://drive.google.com/file/d/1ET7o43qRFV5xiKyw-qByhw0qFQX0OJ5i/view?usp=sharing)/[USR](https://drive.google.com/file/d/1MagXXnjVRdYG-25J8kq3e4o0ts7fcBMS/view?usp=sharing)

BaiduNetdisk: [AISTD](https://pan.baidu.com/s/18fcRpubDixjHpWegIFiU1A)/[ISTD](https://pan.baidu.com/s/1rXnCGbr87Nc3oUGTVu190g)/[USR](https://pan.baidu.com/s/1QtAdumC_jfDfb-iq7bGZ6g) (Access code: 1111)

All codes will be released to public soon.

## AISTD Results (size: 480x640)
| Method | Shadow | Non-shadow | All |
|------|:-----|:-----:|------|
| [Mask-ShadowGAN](https://github.com/xw-hu/Mask-ShadowGAN)(our run)| 11.5* | 4.5* | 5.5 |
| LG-ShadowNet | 10.6* | 4.0* | 5.0 |

## AISTD Results (size: 256x256) 
| Method | Shadow | Non-shadow | All |
|------|:-----|:-----:|------|
| [Mask-ShadowGAN](https://github.com/xw-hu/Mask-ShadowGAN)(our run)| 10.8* | 3.8* | 4.8 |
| LG-ShadowNet | 9.9* | 3.4* | 4.4 |

*Since the ~~RMSE~~ (MAE) in shadow and non-shadow regions are computed on each image first and then compute the average of all images, the results may be different from yours.
