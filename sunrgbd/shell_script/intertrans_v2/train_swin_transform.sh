
# train RGB with  pretrained weights from coco
cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from.py 
python tools/train.py  ${cfg} > train_swin_for_sunrgbd_data_by_rgb_images_with_pretrain_weight_from_coco.log

# Train WITHOUT a pretrained model, train from scratch
#cfg=configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain.py
#python tools/train.py  ${cfg} > train_swin_without_intertrans_new_coco_dataset.log
