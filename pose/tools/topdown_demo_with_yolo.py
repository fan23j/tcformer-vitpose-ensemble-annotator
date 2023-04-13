# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import json_tricks as json
import mmcv
import mmengine
import numpy as np
from tqdm import tqdm
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.models import build_posenet
import tcformer

def main():
    """Visualize the demo images.
    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--img-root', default="")

    args = parser.parse_args()
    
    cfg = Config.fromfile(args.pose_config)

    assert args.show or (args.out_img_root != '')
    if args.out_img_root:
        mmengine.mkdir_or_exist(args.out_img_root)
    if args.save_predictions:
        assert args.out_img_root != ''
        args.pred_save_path = f'{args.out_img_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = torch.hub.load("ultralytics/yolov5", 'yolov5l6')

    # build the model and load checkpoint
    pose_model = build_posenet(cfg.model)
    pose_model = MMDataParallel(pose_model, device_ids=[0])

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(pose_model, args.pose_checkpoint, map_location='cuda')
    pose_model.cfg = cfg
        
    # initalize results containers
    pose_results_list = []
    image_list = []
    
    # run detections
    for img_name in tqdm(os.listdir(args.img_root)):
        image_path = os.path.join(args.img_root, img_name)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        try:
            dets = detector(image_path)
        except Exception as e:
            print(e)
            print(image_path)
            continue
        
        # initialize person detections list
        person_detections = []

        # process results
        df = dets.pandas().xyxy[0]
        df = df[df['name'] == 'person']
        detections = df.iloc[:, :5].values.tolist()
        
        # filter detections results
        for bbox in detections:
            if bbox[4] > args.bbox_thr:
                person_detections.append({'bbox': bbox})

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None
    
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_path,
            person_detections,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{img_name}')

        # save results
        pose_results_list.append(pose_results)
        image_list.append(img_name)
        
        # show the results
        vis_pose_result(
            pose_model,
            image_path,
            pose_results,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)
        
    # write to xml
#     output_xml = generate_cvat_xml(image_list, pose_results_list)
#     with open('output_cvat.xml', 'w') as f:
#         f.write(output_xml)
    # write to json
    results = {}
    results["images"] = image_list
    results["annotations"] = pose_results_list
    
    with open("results.json", "w") as out:
        json.dump(results, out)

if __name__ == '__main__':
    main()
