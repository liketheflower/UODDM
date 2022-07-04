# You Only Need One Detector #  
Related to our paper You Only Need One Detector: Unified Object Detector for Different Modalities based on Vision Transformers.
Our unified model can process RGB images, pseudo images converted from point clouds or inter-modality mixing of RGB image and pseudo images converted from point clouds.   
Comparison of other systems can be seen ![here](./main-yonod-006223_1-2.png)
This repo contains the supported code and configuration files to reproduce object detection results of [simCrossTrans]. It is modified based on [Swin Transformer for object detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). [Original Readme](./README_original.md)
## Usage ##
### Inference###
Run the following shell script:
```bash
./sunrgbd/shell_script/yonod/inference/inference.sh
```
### Installation ##
Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.


