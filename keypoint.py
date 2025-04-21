######
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
########
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker
from lib.utils.demo_utils import convert_crop_coords_to_orig_img
import os

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)



def extract_keypoint(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = x.shape[0]
    num_frames = x.shape[1]
    MIN_NUM_FRAMES = 8#8


    keypoint = []
    for i in range(batch_size):
        print('batch NO.',i)

        image_folder = "tmp_frames"
        os.makedirs(image_folder, exist_ok=True)
        for j in range(num_frames):
            #print(x.shape)
            frame = x[i, j]
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            ###########3
            brightness_factor = 1.2
            frame = frame * brightness_factor

            frame = frame.clip(0.0, 1.0)
            ##########33
            image_path = os.path.join(image_folder, f"frame_{i}_{j}.jpg")
            torchvision.utils.save_image(frame, image_path)

        mot = MPT(
            device=device,
            batch_size=12,  # 8,#16,  # 12
            display=False,  # False
            detector_type='yolo',  # yolo
            output_format='dict',
            yolo_img_size=416,  # 416
        )
        tracking_results = mot(image_folder)
        print('tracking_results', list(tracking_results.keys()))
        for person_id in list(tracking_results.keys()):
            print('MIN_NUM_FRAMES',tracking_results[person_id]['frames'].shape[0])
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]
        model = VIBE_Demo(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(device)

        # ========= Load pretrained weights ========= #
        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        print(f'Loaded pretrained weights from \"{pretrained_file}\"')

        #breakpoint()
        ##########
        if len(list(tracking_results.keys())) == 0:
            smpl_joints2d = torch.tensor([])
        ############

        for person_id in tqdm(list(tracking_results.keys())):
            #print('person_id', person_id)
            bboxes = joints2d = None
            bboxes = tracking_results[person_id]['bbox']
            frames = tracking_results[person_id]['frames']
            bbox_scale = 1.1
            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )
            bboxes = dataset.bboxes
            dataloader = DataLoader(dataset, batch_size=4, num_workers=1)
            '''
            with torch.no_grad():
                smpl_joints2d = []
                for batch in dataloader:
                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)
                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            '''
            with torch.no_grad():
                smpl_joints2d = []
                for batch in dataloader:
                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)
                    batch_size1, seqlen = batch.shape[:2]
                    output = model(batch)[-1]
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size1 * seqlen, -1, 2))  # torch.Size([384, 49, 2])
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
                print('smpl_joints2d',smpl_joints2d.shape)
                del batch

            smpl_joints2d = smpl_joints2d.cpu().numpy()

            joints2d_img_coord = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=224,
            )
            print('joints2d_img_coord', joints2d_img_coord.shape)
            joints2d_img_coord = joints2d_img_coord[:8]
        #print(joints2d_img_coord)
        keypoint.append(joints2d_img_coord)


        shutil.rmtree(image_folder)

    keypoint = np.array(keypoint)


    keypoint = torch.from_numpy(keypoint)
    keypoint = keypoint.to(device)


    return keypoint