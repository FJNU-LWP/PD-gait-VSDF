import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import model
from src import util
from src.body import Body
import torch
import os
def op_extract_keypoint(x):
    body_estimation = Body('op_model/body_pose_model.pth')
    oriImg = cv2.imread("./tmp_frames/1.jpg")
    candidate, subset = body_estimation(oriImg.cpu())
    print(candidate)
    print(candidate.shape)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = x.shape[0]
    num_frames = x.shape[1]


    keypoint = []
    for i in range(batch_size):
        print('batch NO.',i)

        image_folder = "tmp_frames"
        os.makedirs(image_folder, exist_ok=True)
        for j in range(num_frames):
            #print(x.shape)
            frame = x[i, j]
            frame = (frame - frame.min()) / (frame.max() - frame.min())

            print(type(frame))
            print(frame.shape)
            frame_np = frame.cpu().numpy()
            frame_np = frame_np.transpose(1, 2, 0)

            oriImg = cv2.imread("./tmp_frames/1.jpg")
            candidate, subset = body_estimation(oriImg)
            print(candidate)
            print(candidate.shape)
            breakpoint()
            plt.imshow(frame_np)
            plt.show()
            breakpoint()
            candidate, subset = body_estimation(frame)


            ###########3
            brightness_factor = 1.2
            frame = frame * brightness_factor

            frame = frame.clip(0.0, 1.0)
            ##########33
            image_path = os.path.join(image_folder, f"frame_{i}_{j}.jpg")
            torchvision.utils.save_image(frame, image_path)