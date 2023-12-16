import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights

def points_initialization(opt, image):
    '''
    Different options for gaussian initialization: 
        random -- fully random type
        uniform -- uniform grid with 32 steps for one row
        keypoints -- detected keypoints with added random points
    '''
    if opt.initialization_type == 'random':
        points = random_initialization(opt, image)
    elif opt.initialization_type == 'uniform':
        points = uniform_initialization(opt, image)
    elif opt.initialization_type == 'keypoints':
        points = keypoints_initialization(opt, image)
    return points

def random_initialization(opt, np_image):
    points = np.random.randint(0, [np_image.shape[0], np_image.shape[1]], size=(opt.limit_points_number, 2))
    return points

def uniform_initialization(opt, np_image):
    num_steps = 32
    linspace = torch.linspace(0, opt.img_size[0]-1, steps=num_steps, device=opt.device).to(torch.int32)
    xx = linspace[:, None].expand(-1, num_steps)
    yy = linspace[None, :].expand(num_steps, -1)
    points = torch.cat((xx.unsqueeze(-1), yy.unsqueeze(-1)), -1).cpu().numpy().reshape(-1, 2)
    left_size = opt.limit_points_number - opt.start_points_number
    left_points = np.random.randint(0, [np_image.shape[0], np_image.shape[1]], size=(left_size, 2))
    points = np.concatenate((points[:opt.start_points_number], left_points), 0)
    return points

def keypoints_initialization(opt, np_image):
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_LEGACY, 
                                      weights_backbone = ResNet50_Weights.IMAGENET1K_V2
                                     ).to(opt.device)
    model.eval()
    x = torch.tensor(np_image).permute(2,0,1).unsqueeze(0).to(opt.device).float()
    predictions = model(x)
    keypoints = predictions[0]['keypoints'].reshape(-1, 3)[:, :2].detach().cpu().numpy().astype(int)
    left_size = opt.limit_points_number - keypoints.shape[0]
    xy = np.random.randint(0, [np_image.shape[0], np_image.shape[1]], size=(left_size, 2))
    points = np.concatenate((keypoints, xy), 0)
    return points