
# Train both rgb and dhs images at the same time with pretrained weights from COCO dataset based on RGB images
cfg=configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed_with_chess1_test_on_val_chess1.py
echo "train rgb and dhs"
python tools/train.py  ${cfg} > train_swin_s_rgbdhsmixed_with_chess1_100epochs.log 
