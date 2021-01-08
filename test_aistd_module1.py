import argparse
import sys
import os
from os.path import exists, join as join_paths

import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize

from model_module1 import Generator_S2F
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,3,1,2,7,0,6"

parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--generator_A2B', type=str, default='AISTD/Module1/netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()

## ISTD
opt.dataroot_A = '/home/liuzhihao/dataset/ISTD/test/test_A'
opt.im_suf_A = '.png'

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')


netG_A2B = Generator_S2F(opt.input_nc, opt.output_nc)

if opt.cuda:
    netG_A2B.to(device)

netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_A2B.eval()

if not os.path.exists('model_istda_lgsn_module1/B'):
    os.makedirs('model_istda_lgsn_module1/B')

gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

for idx, img_name in enumerate(gt_list):
    print('predicting: %d / %d' % (idx + 1, len(gt_list)))
    # Set model input
    labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)))
    img=labimage[:,:,0]
    h=img.shape[0]
    w=img.shape[1]
    img=resize(img,[480,640])
    img=np.asarray(img)/50.0-1.0
    img=torch.from_numpy(img).float()
    img=img.view(480,640,1)
    img=img.transpose(0, 1).transpose(0, 2).contiguous()
    img_var=img.unsqueeze(0).to(device)

    # Generate output
    temp_B=netG_A2B(img_var)
    fake_B=50.0*(temp_B.data+1.0)
    fake_B=fake_B.data.squeeze(0).cpu()
    fake_B=fake_B.transpose(0, 2).transpose(0, 1).contiguous().numpy()
    fake_B=resize(fake_B,(h,w))
    for i in range(0,h):
        for j in range(0,w):
            labimage[i,j,0]=fake_B[i,j]
    outputimage=color.lab2rgb(labimage)
    
    save_result = join_paths('./model_istda_lgsn_module1/B/%s'% (img_name + opt.im_suf_A))
    io.imsave(save_result,outputimage)
    print('Generated images %04d of %04d' % (idx+1, len(gt_list)))
