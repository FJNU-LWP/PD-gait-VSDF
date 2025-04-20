# PD-gait-VSDF

This project is for the video-based PD gait assassment based on Vision-skeleton Dual-modality Framework.

## Requirements

The code is built with following libraries:

- torch
- torchmetrics
- torchvision
- pytorch-lightning
- pytorchvideo
- scikit-image

## Pose Extraction
In this project, we provide two pose estimation extractor interfaces: OpenPose and VIBE, implemented in `op_keypoint.py` and `VIBE_keypoint.py`, respectively.
## Training
To train the model(s) in the paper, run this command:

### Example Usage

```bash
python main.py
-lr 0.001 -epoch 50 -batch_size 16
-num_class 4 -optim_type 'adam' -objective 'supervised'
-root_dir <train_video_path> -val_data_path <val_video_path>
```

## Access to Datasets

You must contact us first. Follow the [link](#) to apply for our dataset!  
Please ensure that you satisfy the following application requirements:

- Your institution must be non-profit, non-commercial;  
- You must provide proof of relevant medical credentials to show that you are engaged in the same research area;  
- We take the patients' privacy very seriously, even if they have signed a consent form with us. Therefore, you must contact us (email: [QBX20210094@yjs.fjnu.edu.cn](mailto:QBX20210094@yjs.fjnu.edu.cn)) first so that we can confirm and review your information in detail.  
- You will then be required to sign a contract to ensure that you will not make the dataset public and use it for academic research only.

