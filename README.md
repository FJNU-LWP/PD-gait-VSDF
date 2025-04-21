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
These resources were used or cited within the code:
- [VIBE](https://github.com/mkocabas/VIBE)
- [Openpose API](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Training and Evaluating
To train the model(s) in the paper, run this command:

```bash
python main.py
-lr 0.001 -epoch 50 -batch_size 16
-num_class 4 -optim_type 'adam' -objective 'supervised'
-root_dir <train_video_path> -val_data_path <val_video_path>
```
## License

The provided project code is licensed for non-commercial research purposes only. Any commercial use, redistribution, or modification of the code for profit-making activities is strictly prohibited.Users of this code must comply with all relevant research ethics regulations and institutional review requirements.
