from os.path import join, isfile, isdir
from os import listdir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
from PIL import Image
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch
import time
from options.train_options import TrainOptions, TestOptions
from models import create_model
from util.visualizer import Visualizer
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm
from fusion_dataset import Fusion_Testing_Dataset
from util import util
import multiprocessing
multiprocessing.set_start_method('spawn', True)
torch.backends.cudnn.benchmark = True
from image_util import *

# set test options
opt = TestOptions().parse()
save_img_path = opt.results_img_dir
if os.path.isdir(save_img_path) is False:
    print('Create path: {0}'.format(save_img_path))
    os.makedirs(save_img_path)
opt.batch_size = 1

# create instance segmentation predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# create colorization model
model = create_model(opt)
# model.setup_to_test('coco_finetuned_mask_256')
model.setup_to_test('coco_finetuned_mask_256_ffs')

input_dir = opt.test_img_dir
image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
print('#Testing images = %d' % len(image_list))

transforms = transforms.Compose([transforms.Resize((opt.fineSize, opt.fineSize), interpolation=2), transforms.ToTensor()])
for image_path in tqdm(image_list, dynamic_ncols=True):
    img = cv2.imread(join(input_dir, image_path))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
    outputs = predictor(l_stack)

    # get the bounding box with the [box_num_upbound] highest score
    box_num_upbound = 8
    pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy().astype(np.int32)
    if box_num_upbound > 0 and pred_bbox.shape[0] > box_num_upbound:
        pred_scores = outputs["instances"].scores.cpu().data.numpy()
        index_mask = np.argsort(pred_scores, axis=0)[pred_scores.shape[0] - box_num_upbound: pred_scores.shape[0]]
        pred_bbox = pred_bbox[index_mask]

    # process full image
    img_list = []
    pil_img = Image.open(join(input_dir, image_path))
    if len(np.asarray(pil_img).shape) == 2:
        pil_img = np.stack([np.asarray(pil_img), np.asarray(pil_img), np.asarray(pil_img)], 2)
        pil_img = Image.fromarray(pil_img)
    img_list.append(transforms(pil_img))

    # process cropped image
    cropped_img_list = []
    index_list = range(len(pred_bbox))
    box_info, box_info_2x, box_info_4x, box_info_8x = np.zeros((4, len(index_list), 6))
    for i in index_list:
        startx, starty, endx, endy = pred_bbox[i]
        box_info[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, opt.fineSize))
        box_info_2x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, opt.fineSize // 2))
        box_info_4x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, opt.fineSize // 4))
        box_info_8x[i] = np.array(get_box_info(pred_bbox[i], pil_img.size, opt.fineSize // 8))
        cropped_img = transforms(pil_img.crop((startx, starty, endx, endy)))
        cropped_img_list.append(cropped_img)

    output = {}
    output['full_img'] = torch.stack(img_list)
    output['file_id'] = image_path.split('.')[0]
    if len(pred_bbox) > 0:
        output['cropped_img'] = torch.stack(cropped_img_list)
        output['box_info'] = torch.from_numpy(box_info).type(torch.long)
        output['box_info_2x'] = torch.from_numpy(box_info_2x).type(torch.long)
        output['box_info_4x'] = torch.from_numpy(box_info_4x).type(torch.long)
        output['box_info_8x'] = torch.from_numpy(box_info_8x).type(torch.long)
        output['empty_box'] = False
    else:
        output['empty_box'] = True

    count_empty = 0

    # if os.path.isfile(join(save_img_path, output['file_id'][0] + '.png')) is True:
    #     continue
    output['full_img'] = output['full_img'].cuda()
    if output['empty_box'] == 0:
        output['cropped_img'] = output['cropped_img'].cuda()
        box_info = output['box_info']
        box_info_2x = output['box_info_2x']
        box_info_4x = output['box_info_4x']
        box_info_8x = output['box_info_8x']
        output['cropped_img'] = output['cropped_img'].unsqueeze(0)
        output['full_img'] = output['full_img'].unsqueeze(0)
        cropped_data = util.get_colorization_data(output['cropped_img'], opt, ab_thresh=0, p=opt.sample_p)
        full_img_data = util.get_colorization_data(output['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_input(cropped_data)
        model.set_fusion_input(full_img_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
        model.forward()
    else:
        count_empty += 1
        full_img_data = util.get_colorization_data(output['full_img'], opt, ab_thresh=0, p=opt.sample_p)
        model.set_forward_without_box(full_img_data)
    model.save_current_imgs(join(save_img_path, output['file_id'] + '.png'))
print('{0} images without bounding boxes'.format(count_empty))
