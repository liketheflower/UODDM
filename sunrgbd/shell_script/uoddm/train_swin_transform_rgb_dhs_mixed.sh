
# Train both rgb and dhs images at the same time with pretrained weights from COCO dataset based on RGB images
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed.py 
echo "train rgb and dhs"
python tools/train.py  ${cfg} > train_swin_rgb2dhs_50epochs.log 
