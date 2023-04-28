
# Train both rgb and dhs images at the same time with pretrained weights from COCO dataset based on RGB images
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed_with_chess1.py 
echo "train rgb and dhs"
python tools/train.py  ${cfg} > train_swin_rgbdhs_with_chess1_200epochs.log 
