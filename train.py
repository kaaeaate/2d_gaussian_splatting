import torch
from torch.optim import Adam
import os
import argparse
from tqdm import trange

from scene.gaussian_model import get_gaussian_image
from utils.loss_utils import l1_dssim_loss
from utils.training_utils import *
from utils.validation_utils import psnr, get_video
from utils.preprocessing import initialize_tensors, make_parameter
from lpipsPyTorch import lpips

import warnings
warnings.filterwarnings("ignore")


def train(args, gt_image, cov_matrix_tensor, rgba_matrix_tensor, points_locs_rensor, current_points_places, output_path):
    optimizer = Adam([cov_matrix_tensor, rgba_matrix_tensor, points_locs_rensor], lr=args.learning_rate)
    
    losses_lst = []
    last_added_point_idx = args.start_points_number
    for iteration in trange(args.first_iter, args.iterations+1):
        local_cov_m = cov_matrix_tensor[current_points_places]
        local_rgba = rgba_matrix_tensor[current_points_places]
        local_points = points_locs_rensor[current_points_places]     

        gaussian_image = get_gaussian_image(args, local_cov_m, local_rgba, local_points)
        loss = l1_dssim_loss(gaussian_image, gt_image, lambda_dssim=0.2)

        optimizer.zero_grad()
        loss.backward()

        # Prunning
        if iteration % args.refinement_iter == 0:
            (cov_matrix_tensor, rgba_matrix_tensor, 
             points_locs_rensor, current_points_places) = remove_gaussians(cov_matrix_tensor, rgba_matrix_tensor, 
                                                                           points_locs_rensor, current_points_places)
            reversed_places = torch.logical_not(current_points_places)
            cov_matrix_tensor.grad.data[reversed_places] = 0
            rgba_matrix_tensor.grad.data[reversed_places] = 0
            points_locs_rensor.grad.data[reversed_places] = 0

        # Densification
        if iteration % args.densification_iter == 0 and iteration > 0 and len(local_points) < args.limit_points_number:
            splitting_idxs, clonning_idxs = get_idxs_for_densification(args, cov_matrix_tensor, rgba_matrix_tensor, 
                                                                       points_locs_rensor, current_points_places)
            # Over-reconstruction
            if len(splitting_idxs) > 0:
                (current_points_places, cov_matrix_tensor, 
                 rgba_matrix_tensor, points_locs_rensor, last_added_point_idx) = split_points(args, current_points_places, 
                                                                                              cov_matrix_tensor, rgba_matrix_tensor, 
                                                                                              points_locs_rensor, last_added_point_idx,
                                                                                              splitting_idxs)
            # Under-reconstruction    
            if len(clonning_idxs) > 0:
                (current_points_places, cov_matrix_tensor, 
                 rgba_matrix_tensor, points_locs_rensor, last_added_point_idx) = clone_points(args, current_points_places, 
                                                                                              cov_matrix_tensor, rgba_matrix_tensor, 
                                                                                              points_locs_rensor, last_added_point_idx, 
                                                                                              clonning_idxs)

        optimizer.step()

        loss = loss.item()
        losses_lst.append(loss)    
        torch.cuda.empty_cache()    
        
        if iteration % 10 == 0 and iteration < 200: 
            visualize_result(gaussian_image, gt_image, f'{output_path}/images/iter_{iteration}.jpg', len(local_points))
        if iteration % args.logging_iter == 0:        
            lpips_val = lpips(gaussian_image.permute(2,0,1), gt_image.permute(2,0,1)).detach().item()
            psnt_val = psnr(gaussian_image.permute(2,0,1), gt_image.permute(2,0,1)).mean().detach().item()           
            logging_line = f' Iteration: {iteration}, Loss: {loss:.4f}, PSNR: {psnt_val:.4f}, LPIPS: {lpips_val:.4f}'
            print(logging_line)
            with open(f'{output_path}/logs.txt', 'a') as f:
                f.write(logging_line)
                f.write('\n')
            visualize_result(gaussian_image, gt_image, f'{output_path}/images/iter_{iteration}.jpg', len(local_points), dpi=100)
            del psnt_val
        del loss
        del gaussian_image

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--first_iter', default=0, type=int)
    parser.add_argument('--iterations', default=5000, type=int)
    parser.add_argument('--refinement_iter', default=1, type=int)
    parser.add_argument('--densification_iter', default=250, type=int)
    parser.add_argument('--logging_iter', default=100, type=int)    
    parser.add_argument('--img_size', default=(256, 256), type=tuple)
    parser.add_argument('--gaussian_init_scale', '-gs', default=5, type=int)
    
    parser.add_argument('--scale_factor', default=1.6, type=float)
    parser.add_argument('--start_points_number', default=1000, type=int)
    parser.add_argument('--limit_points_number', default=10000, type=int)
    parser.add_argument('--grad_threshold', default=0.002, type=float)
    parser.add_argument('--gauss_threshold', default=0.05, type=float)
    
    parser.add_argument('--learning_rate', '-lr', default=0.01, type=float)
    parser.add_argument('--lambda', default=0.2, type=float)
    
    parser.add_argument('--image_path', default='images/mikki.jpg', type=str)
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--get_video', default=False, type=bool)
    parser.add_argument('--initialization_type', default='random', type=str, choices=['random', 'uniform', 'keypoints'])
    args, _ = parser.parse_known_args()
    args.device = 'cuda'
           
    if args.output_folder is None:
        exp_name = args.image_path.split('/')[-1].split('.')[0] + '_' + str(args.img_size[0])
    else:
        exp_name = args.output_folder
    output_path = f'experiments/{exp_name}'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f'{output_path}/images', exist_ok=True)
    
    gt_image, current_points_places, scaling, rotation, colors, alphas, points_locations = initialize_tensors(args)
    cov_matrix = torch.cat([scaling, rotation], dim=1)
    rgba_matrix = torch.cat([alphas, colors], dim=1)
    cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor = make_parameter([cov_matrix, rgba_matrix, points_locations])       
    train(args, gt_image, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, current_points_places, output_path)
    
    if args.get_video:
        get_video(output_path, video_name='test_video')
    