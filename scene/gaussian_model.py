import torch
import torch.nn.functional as F
import numpy as np
from utils.general_utils import build_covariance_from_scaling_rotation

def get_covariance(opt, scaling, rotation, n_points):
    scaling_modifier = 1
    cov_mat = build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, opt.device)
    det = cov_mat[:, 0, 0]*cov_mat[:, 1, 1] - cov_mat[:, 1, 0]*cov_mat[:, 0, 1]
    if (det <= 0).any():
        raise Exception('Covariance matrix is not positive semi-definite')
    return cov_mat

def get_normal_pdf(opt, gaussian_size, S, n_points):
    num_steps = gaussian_size
    linspace = torch.linspace(-4, 4, steps=num_steps, device=opt.device)
    xx = linspace[None, :, None].expand(-1, -1, num_steps)
    yy = linspace[None, None, :].expand(-1, num_steps, -1)
    xy = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), -1)
    inv_S = torch.inverse(S)    
    gauss = torch.exp(torch.stack([-0.5 * xy[:,:,:,i] * inv_S[:,i,j][:, None, None] * xy[:,:,:,j]
                                   for i in range(2) for j in range(2)]).sum(dim=0))
    normal_pdf = gauss/ (2 * torch.tensor(np.pi, device=opt.device) * torch.sqrt(torch.det(S)).view(n_points, 1, 1)) # normal pdf

    pdf_max = torch.amax(normal_pdf, dim=(-1, -2), keepdim=True)
    normal_pdf = normal_pdf / pdf_max
    channeled_normal_pdf = normal_pdf.repeat(1, 3, 1)
    channeled_normal_pdf = channeled_normal_pdf.view(3*n_points, num_steps, num_steps)[None, :] #  (3*n_points, ks, ks)
    pdf_img = channeled_normal_pdf.reshape(n_points, 3, num_steps, num_steps) #(n_points, 3, ks, ks)
    return pdf_img

def pad_image(start_img_size, final_img_size, img):
    padding_1 = final_img_size[0] - start_img_size[0]
    padding_2 = final_img_size[1] - start_img_size[1] 
    padding = (int(padding_2 / 2), int(padding_2 / 2) + padding_2 % 2,
               int(padding_1 / 2), int(padding_1 / 2) + padding_1 % 2)
    image_padded = F.pad(img, padding, "constant", 0)
    return image_padded

def put_gaussians_to_points(opt, pdf_img, n_points, points):
    theta = torch.eye(3, 3).to(opt.device).repeat(n_points, 1,1)
    theta[:, :2, 2] = points
    theta = theta[:,:2, :]
    grid = F.affine_grid(theta, size=(n_points, 3, opt.img_size[0], opt.img_size[1]), align_corners=True)
    gaussians_to_points = F.grid_sample(pdf_img, grid, align_corners=True)
    return gaussians_to_points

def get_gaussian_image(opt, cov_matrix, rgba, points):  
    scaling, rotation = torch.sigmoid(cov_matrix[:, :2]), cov_matrix[:, 2:3]
    colors, alpha = torch.sigmoid(rgba[:, 1:4]), torch.sigmoid(rgba[:, 0])    
    n_points = points.shape[0]
    
    colours_weighted  = colors * alpha.view(n_points, 1)
    gaussian_size = int(opt.img_size[0] / opt.gaussian_init_scale)
    if gaussian_size > opt.img_size[0]:
        raise Exception('Gaussian size is more than image size')
    S = get_covariance(opt, scaling, rotation, n_points)    
    pdf_img = get_normal_pdf(opt, gaussian_size, S, n_points)   
    pdf_img_resized = pad_image(pdf_img.shape[-2:], opt.img_size, pdf_img)
    gaussians_result = put_gaussians_to_points(opt, pdf_img_resized, n_points, points)
    gaussians_blended = gaussians_result * colours_weighted[:, :, None, None]
    gaussian_image = gaussians_blended.permute(0,2,3,1).sum(dim=0).clamp(0, 1)
    return gaussian_image

