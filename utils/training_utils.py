import torch
import matplotlib.pyplot as plt


def remove_gaussians(cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, current_points_places, thresh_alpha=0.01):
    '''
    Remove gaussians by alpha threshold
    Input:
        cov_matrix_tensor: vector consisting of scale and rotation, torch.Size([10000, 3])
        rgba_matrix_tensor: RGB channels and alpha, torch.Size([10000, 4])
        points_locs_tensor: gaussians locations, torch.Size([10000, 2])
        current_points_places: bool mask for places where gaussians are located currently, torch.Size([10000])
        thresh_alpha: threshold, float
    Output:
        Masked input tensors (without points corresponding aplha < threshold)
    '''
    alpha = torch.sigmoid(rgba_matrix_tensor[:, 0])
    removed_gaussians_idxs = (alpha < thresh_alpha).nonzero()[:,0]
    if len(removed_gaussians_idxs) >= 1:
        current_points_places[removed_gaussians_idxs] = False
        reversed_places = torch.logical_not(current_points_places)
        cov_matrix_tensor.data[reversed_places] = 0
        rgba_matrix_tensor.data[reversed_places] = 0
        points_locs_tensor.data[reversed_places] = 0
        print(f'Prunning: {len(removed_gaussians_idxs)} points')
    return cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, current_points_places

def find_idxs_with_big_values(tensor, thresh):
    '''
    Choose indexes of tensor points, which are more than threshold. It is used for points gradients and covariance matrix 
    Input:
        tensor: points gradients or covariance matrix
        thresh: threshold, float
    Output:
        Indexes of tensor points, which are more than threshold
    '''
    tensor_norm = torch.norm(tensor, dim=1, p=2)
    values, idxs = torch.sort(tensor_norm, descending=True)
    values_by_thresh_mask = (values > thresh)
    idxs_by_thresh = idxs[values_by_thresh_mask]
    return idxs_by_thresh

def get_idxs_for_densification(opt, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, current_points_places):
    '''
    Get indexes of points which satisfies condition for densification (Algorithm 1) 
    Input:
        cov_matrix_tensor: vector consisting of scale and rotation, torch.Size([10000, 3])
        rgba_matrix_tensor: RGB channels and alpha, torch.Size([10000, 4])
        points_locs_tensor: gaussians locations, torch.Size([10000, 2])
        current_points_places: bool mask for places where gaussians are located currently, torch.Size([10000])
    Output:
        Indexes for splitting and clonning
    '''
    # Get values by mask (current_points_places), where gaussians are located
    grads_idxs = find_idxs_with_big_values(points_locs_tensor.grad[current_points_places], opt.grad_threshold)
    covmat_idxs = find_idxs_with_big_values(torch.sigmoid(cov_matrix_tensor.data[current_points_places][:, 0:2]), opt.gauss_threshold)

    intersection_idxs = torch.isin(grads_idxs, covmat_idxs)
    splitting_idxs = grads_idxs[intersection_idxs]
    clonning_indices = grads_idxs[~intersection_idxs]
    return splitting_idxs, clonning_indices

def add_points_in_space(opt, current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, 
                        last_idx, target_idxs, regime='splitting'):
    '''
    Implementation of densification.  
    Input:
        cov_matrix_tensor: vector consisting of scale and rotation, torch.Size([10000, 3])
        rgba_matrix_tensor: RGB channels and alpha, torch.Size([10000, 4])
        points_locs_tensor: gaussians locations, torch.Size([10000, 2])
        current_points_places: bool mask for places where gaussians are located currently, torch.Size([10000])
        last_idx: index of point that was last added in points list, float
        target_idxs: indexes of gaussians for adding (splitting or clonning), torch.Tensor
    Output:
        Masked input tensors with new added gaussians
    '''
    num_points = len(target_idxs)
    current_points_places[last_idx+1:last_idx+num_points+1] = True
    cov_matrix_tensor.data[last_idx:last_idx+num_points, :] = cov_matrix_tensor.data[target_idxs, :]
    rgba_matrix_tensor.data[last_idx:last_idx+num_points, :] = rgba_matrix_tensor.data[target_idxs, :]
    points_locs_tensor.data[last_idx:last_idx+num_points, :] = points_locs_tensor.data[target_idxs, :]
    
    # Apply scale factor to splitting gaussians
    if regime == 'splitting':
        additional_idxs = torch.tensor([i for i in range(last_idx, last_idx+len(target_idxs)+1)]).to(opt.device)
        resulted_idxs = torch.cat((target_idxs, additional_idxs), 0)
        cov_matrix_tensor.data[resulted_idxs, 0:2] /= opt.scale_factor
    return current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor 

def split_points(opt, current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor,
                 last_added_point_idx, splitting_idxs, regime='splitting'):
    '''
    Implementation of densification -- splitting points (over-reconstruction). 
    For more details see description for add_points_in_space.
    '''
    (current_points_places, cov_matrix_tensor, 
     rgba_matrix_tensor, points_locs_tensor) = add_points_in_space(opt, current_points_places, cov_matrix_tensor, rgba_matrix_tensor,
                                                                   points_locs_tensor, last_added_point_idx, 
                                                                   splitting_idxs, regime=regime)  
    last_added_point_idx = last_added_point_idx + len(splitting_idxs)
    return current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, last_added_point_idx

def clone_points(opt, current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, 
                 last_added_point_idx, clonning_idxs, regime='clonning'):
    '''
    Implementation of densification -- clonning points (under-reconstruction). 
    For more details see description for add_points_in_space.
    '''
    (current_points_places, cov_matrix_tensor, 
     rgba_matrix_tensor, points_locs_tensor) = add_points_in_space(opt, current_points_places, cov_matrix_tensor, rgba_matrix_tensor,
                                                                   points_locs_tensor, last_added_point_idx, 
                                                                   clonning_idxs, regime='clonning')  
    last_added_point_idx = last_added_point_idx + len(clonning_idxs)
    return current_points_places, cov_matrix_tensor, rgba_matrix_tensor, points_locs_tensor, last_added_point_idx

def visualize_result(predict, gt, folder_path, num_points, dpi=80):
    plt.ioff()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt.cpu().detach().numpy())
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(predict.cpu().detach().numpy())
    ax[1].set_title(f'2D GS result: {num_points} gaussians')
    ax[1].axis('off')
    fig.savefig(folder_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    