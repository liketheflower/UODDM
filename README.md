# You Only Need One Detector #  
Related to our paper You Only Need One Detector: Unified Object Detector for Different Modalities based on Vision Transformers.
Our unified model can process RGB images, pseudo images converted from point clouds or inter-modality mixing of RGB image and pseudo images converted from point clouds.   
Comparison of other systems can be seen ![here](./main-yonod-006223_1-2.png)
This repo contains the supported code and configuration files to reproduce object detection results of [simCrossTrans]. It is modified based on [Swin Transformer for object detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). [Original Readme](./README_original.md)


## MODELs ## 
### Pretained based on COCO dataset ###

Those are the base model used for the YONOD work. You can download them from the official SWIN transformer repo, you can also download a backup from the link(Google Drive) provided here:
| Finetune dataset | model         | checkpoint   | 
|--------------|-----------|------------|
| COCO | Swin-T| [swin-t-model](https://drive.google.com/file/d/1kJ_4Bc2qh7mZdG3E_QhtqMNiOZb-Dj-l/view?usp=sharing)|
| COCO | Swin-S| [swin-s-model](https://drive.google.com/file/d/1tggozECuNp8_Jcj3fKKKq_npQy78EH2X/view?usp=sharing)|

### YONOD finetune on SUN RGBD dataset ###
The YONOD work was finetuning the above model based on SUN RGBD dataset. It has two models based on different modalities:
- INPUT A: RGB.
- INPUT B: RGB and DHS and RGB DHS mixed based on chessboard mixture.    

We also had a input only as DHS model can be found in the [simCrossTrans](https://arxiv.org/abs/2203.10456) work. Here the performance based on mAP50 for SUNRGBD10, which includes a 10 common categories. Details please check the [YONOD](https://arxiv.org/pdf/2207.01071.pdf) paper.

| Finetune dataset |input| model         | checkpoint   | configure file| performance on RGB validation | performance on DHS validation |performance when both RGB and DHS are available|log| 
|--------------|-----------|------------|--------|-------|---------|--------|-----|------| 
|SUN RGBD | INPUT A| swin-t|[basedRGB](https://drive.google.com/file/d/1cfIxRG4vumIAX3T1cem79gqUZcp1o1-n/view?usp=sharing)|[cfg](https://github.com/liketheflower/YONOD/blob/master/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py)|53.9|N/A|N/A|[log](https://raw.githubusercontent.com/liketheflower/YONOD/update_readme/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/20220211_002440.log)|
|SUN RGBD | INPUT B| swin-t|[basedRGBandDHSandRGBDHSmixed](https://drive.google.com/file/d/1cfIxRG4vumIAX3T1cem79gqUZcp1o1-n/view?usp=sharing)|[cfg](https://github.com/liketheflower/YONOD/blob/master/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed_with_chess1/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed_with_chess1.py)|54.2|55.8|58.1|[test_on_RGB](https://github.com/liketheflower/YONOD/blob/master/test_logs_unshow/rgb_dhs_mixed_with_chess1_on_rgb_on_val_chess1/test_swin_rgb_dhs_mixed_with_chess1_yonod_on_rgb_epoch_99.log) [test on RGB DHS mixed](https://raw.githubusercontent.com/liketheflower/YONOD/master/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed_with_chess1/20220510_174307.log)|

## Usage ##
### Finetune based on COCO for SUNRGBD ### 
The sun rgbd dataset training and test can be found in the sunrgbd folder, if you want to train the sunrgbd dataset based on pretrained model on COCO, please do the following:
```bash
cd sunrgbd
./shell_script/yonod/train_swin_transform.sh
```
You need download a pretrained model from the COCO dataset and you can find the models in the MODEL session.
If you want to train a RGB image, please use:# train RGB with pretrained weights from coco for 100 epochs"

### About SUN RGBD categories ###
The SUN RGBD dataset also has 80 categories to align with COCO dataset. The SUNRGBD is direclty overwritten the COCO dataset's class, see [this line](https://github.com/liketheflower/mmdetection_beta/blob/44936c7df0982fbd3cf60cf60f6ad001d56d7019/mmdet/datasets/coco.py#L37)
If you want to directly use the pretrained model from SUN RGBD dataset, you need use the following customized mmdetection (updating the categories name to SUN RGBD and add some inference code). 
```bash
https://github.com/liketheflower/mmdetection_beta
```

### Inference ###
Run the following shell script:
```bash
./sunrgbd/shell_script/yonod/inference/inference.sh
```
### Installation ##
Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.
### Citation ###
Please cite our paper if you feel the repo is useful:
```
@misc{https://doi.org/10.48550/arxiv.2207.01071,
  doi = {10.48550/ARXIV.2207.01071},
  
  url = {https://arxiv.org/abs/2207.01071},
  
  author = {Shen, Xiaoke and Li, Zhujun and Canizales, Jaime and Stamos, Ioannis},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {You Only Need One Detector: Unified Object Detector for Different Modalities based on Vision Transformers},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
