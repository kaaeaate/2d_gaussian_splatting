import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from utils.initialization import points_initialization

def normalize_points(args, points, middle_p=0.5):
    '''
    Normalize points locations to [-1, 1].
    '''
    middle_points = torch.tensor([middle_p, middle_p])
    points = torch.tensor(points / [args.img_size[0], args.img_size[1]])
    normalized_points = -2 * (points - middle_points)
    return normalized_points.float().to(args.device) 

def get_colors(args, image, points):
    '''
    Put colors from target image to initialized points locations.
    '''
    colors = torch.tensor([image[point[1], point[0]] for point in points])
    return colors.float().to(args.device)

def read_image(args):
    image = Image.open(args.image_path).convert('RGB')
    if image.size[0] != image.size[1]:
        raise Exception('Please load image with equal width and height')
    image = image.resize((args.img_size[0], args.img_size[1]))
    image = np.array(image) / 255.0
    return image

def initialize_tensors(args):
    '''
    Get all tensors for gaussians building and optimization.
    '''
    image = read_image(args)    
    points_places = torch.ones(args.start_points_number).bool()
    points_free = torch.zeros(args.limit_points_number - args.start_points_number).bool()
    current_points_places = torch.cat([points_places, points_free], dim=0)
    
    scaling = torch.rand(args.limit_points_number, 2, device=args.device)
    rotation = torch.rand(args.limit_points_number, 1, device=args.device)
    alphas = torch.ones(args.limit_points_number, 1, device=args.device)
    points_locations = points_initialization(args, image)
    colors = get_colors(args, image, points_locations)
    points_locations = normalize_points(args, points_locations)
    gt_image = torch.tensor(image).float().to(args.device)    
    return gt_image, current_points_places, scaling, rotation, colors, alphas, points_locations

def make_parameter(lst_tensors):
    lst_parameters = []
    for tensor in lst_tensors:
        lst_parameters.append(nn.Parameter(tensor))
    return lst_parameters