<<comment
# train RGB with  pretrained weights from coco for 100 epochs
#cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py 
#python tools/train.py  ${cfg} > train_swin_for_sunrgbd_data_by_rgb_images_with_pretrain_weight_from_coco.log

# Train DHS with pretrained weights from RGB of sunrgbd dataset based on the previous RGB 100 epoch checkpoint and train another 50 epochs
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs.py 
python tools/train.py  ${cfg} > train_swin_rgb2dhs.log 
comment
<<comment
# train RGB with  pretrained weights from coco for 50 epoch only
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py 
echo "train rgb"
python tools/train.py  ${cfg} > train_swin_for_sunrgbd_data_by_rgb_images_with_pretrain_weight_from_coco_50epochs.log
comment

# Train DHS with pretrained weights from RGB of sunrgbd dataset based on the previous RGB 50 epoch and train another 50 epochs
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs.py 
echo "train dhs"
python tools/train.py  ${cfg} > train_swin_rgb2dhs_50epochs.log 
