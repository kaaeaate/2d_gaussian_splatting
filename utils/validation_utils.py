import torch
import os
import cv2
import numpy as np
from PIL import Image

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_video(exp_path, video_name='video'):
    lst_final_imgs = os.listdir(exp_path)
    lst_final_imgs = [int(img.split('_')[1].split('.')[0]) for img in lst_final_imgs]
    lst_final_imgs = sorted(lst_final_imgs)
    names = [f'iter_{n}.jpg' for n in lst_final_imgs]
    lst_imgs = []
    for name in names:
        img = Image.open(exp_path + name)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lst_imgs.append(img)

    out = cv2.VideoWriter(f'{video_name}.mp4', 
                          cv2.VideoWriter_fourcc(*'mp4v'), 2, (img.shape[1], img.shape[0]))
    for i in lst_imgs:
        out.write(i)
    out.release()