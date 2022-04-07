# Test swin with rgb2dhs on RGB images
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/epoch_69.pth
python tools/test.py  ${cfg} ${checkpoint} --eval=bbox  > ./test_logs_unshow/test_swin_rgb_yonod_at_epoch100_unshow.log

